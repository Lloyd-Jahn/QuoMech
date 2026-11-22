// High-throughput data-driven float64 stiff ODE engine (ROS2 with analytic Jacobian,
// exact active-species/reaction compaction) and bounded Levenberg-Marquardt fitter
// with OpenMP parallelism. 100% network-agnostic.

#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <atomic>
#include <algorithm>
#include <vector>
#include <omp.h>

namespace {

constexpr int MAX_S = 32;
constexpr int MAX_R = 64;
constexpr int MAX_P = 32;
constexpr double GAMMA_ROS2 = 0.29289321881345247560;  // 1 - 1/sqrt(2), L-stable

struct NetworkDesc {
    int n_species;
    int n_reactions;
    int n_params;
    const int* reactant1;
    const int* reactant2;
    const int* rxn_param_slot;
    int stoich_offset[MAX_R + 1];
    int stoich_species[MAX_R * 6];
    double stoich_delta[MAX_R * 6];
};

// Compact active subsystem for a single mechanism (100% mathematically identical to full system)
struct CompactSubsys {
    int n_as;                       // active species count (typically 6..8 vs 11)
    int n_ar;                       // active reaction count (typically 4..9 vs 17)
    int sub_to_full_s[MAX_S];       // [n_as] -> full species idx
    int ar_full_idx[MAX_R];         // [n_ar] -> full reaction idx
    int ar_r1[MAX_R];               // [n_ar] -> compact species idx
    int ar_r2[MAX_R];               // [n_ar] -> compact species idx (-1 if unimol)
    int ar_stoich_off[MAX_R + 1];
    int ar_stoich_s[MAX_R * 6];     // compact species idx
    double ar_stoich_d[MAX_R * 6];
    double ar_k[MAX_R];             // active rate constants
};

inline void build_network_desc(
    int n_species,
    int n_reactions,
    int n_params,
    const int* reactant1,
    const int* reactant2,
    const int* rxn_param_slot,
    const double* stoich_matrix,
    NetworkDesc& net
) {
    net.n_species = n_species;
    net.n_reactions = n_reactions;
    net.n_params = n_params;
    net.reactant1 = reactant1;
    net.reactant2 = reactant2;
    net.rxn_param_slot = rxn_param_slot;
    int pos = 0;
    for (int r = 0; r < n_reactions; ++r) {
        net.stoich_offset[r] = pos;
        for (int s = 0; s < n_species; ++s) {
            double v = stoich_matrix[s * n_reactions + r];
            if (std::fabs(v) > 1e-15) {
                net.stoich_species[pos] = s;
                net.stoich_delta[pos] = v;
                ++pos;
            }
        }
    }
    net.stoich_offset[n_reactions] = pos;
}

inline bool build_compact_subsys(
    const NetworkDesc& net,
    const int* rxn_mask,
    const double* log10_k,
    CompactSubsys& sub
) {
    const int ns = net.n_species;
    const int nr = net.n_reactions;
    int full_to_sub[MAX_S];
    for (int s = 0; s < ns; ++s) full_to_sub[s] = -1;

    sub.n_as = 0;
    sub.n_ar = 0;
    int spos = 0;

    auto map_s = [&](int full_s) -> int {
        int c = full_to_sub[full_s];
        if (c < 0) {
            c = sub.n_as++;
            full_to_sub[full_s] = c;
            sub.sub_to_full_s[c] = full_s;
        }
        return c;
    };

    for (int r = 0; r < nr; ++r) {
        if (!rxn_mask[r]) continue;
        double lk = log10_k[net.rxn_param_slot[r]];
        if (!std::isfinite(lk) || lk > 18.0) return false;
        int ar = sub.n_ar++;
        sub.ar_full_idx[ar] = r;
        sub.ar_k[ar] = std::pow(10.0, lk);
        sub.ar_r1[ar] = map_s(net.reactant1[r]);
        sub.ar_r2[ar] = (net.reactant2[r] < 0) ? -1 : map_s(net.reactant2[r]);
        sub.ar_stoich_off[ar] = spos;
        for (int idx = net.stoich_offset[r]; idx < net.stoich_offset[r + 1]; ++idx) {
            sub.ar_stoich_s[spos] = map_s(net.stoich_species[idx]);
            sub.ar_stoich_d[spos] = net.stoich_delta[idx];
            ++spos;
        }
    }
    sub.ar_stoich_off[sub.n_ar] = spos;
    return true;
}

inline void eval_rhs_compact(
    const CompactSubsys& sub,
    const double* y,
    double* f,
    double* rates
) {
    const int ns = sub.n_as;
    const int nr = sub.n_ar;
    for (int s = 0; s < ns; ++s) f[s] = 0.0;

    for (int r = 0; r < nr; ++r) {
        const double kr = sub.ar_k[r];
        const int a = sub.ar_r1[r];
        const int b = sub.ar_r2[r];
        const double v = (b < 0) ? (kr * y[a]) : (kr * y[a] * y[b]);
        if (rates) rates[r] = v;
        const int beg = sub.ar_stoich_off[r];
        const int end = sub.ar_stoich_off[r + 1];
        for (int idx = beg; idx < end; ++idx) {
            f[sub.ar_stoich_s[idx]] += sub.ar_stoich_d[idx] * v;
        }
    }
}

inline void eval_rhs_jac_compact(
    const CompactSubsys& sub,
    const double* y,
    double* f,
    double* J,
    double* rates
) {
    const int ns = sub.n_as;
    const int nr = sub.n_ar;
    for (int s = 0; s < ns; ++s) f[s] = 0.0;
    std::memset(J, 0, sizeof(double) * ns * ns);

    for (int r = 0; r < nr; ++r) {
        const double kr = sub.ar_k[r];
        const int a = sub.ar_r1[r];
        const int b = sub.ar_r2[r];
        double v, dv_da, dv_db = 0.0;
        if (b < 0) {
            v = kr * y[a];
            dv_da = kr;
        } else if (a == b) {
            v = kr * y[a] * y[a];
            dv_da = 2.0 * kr * y[a];
        } else {
            v = kr * y[a] * y[b];
            dv_da = kr * y[b];
            dv_db = kr * y[a];
        }
        if (rates) rates[r] = v;

        const int beg = sub.ar_stoich_off[r];
        const int end = sub.ar_stoich_off[r + 1];
        if (b < 0 || a == b) {
            for (int idx = beg; idx < end; ++idx) {
                const int s = sub.ar_stoich_s[idx];
                const double nu = sub.ar_stoich_d[idx];
                f[s] += nu * v;
                J[s * ns + a] += nu * dv_da;
            }
        } else {
            for (int idx = beg; idx < end; ++idx) {
                const int s = sub.ar_stoich_s[idx];
                const double nu = sub.ar_stoich_d[idx];
                f[s] += nu * v;
                J[s * ns + a] += nu * dv_da;
                J[s * ns + b] += nu * dv_db;
            }
        }
    }
}

inline bool lu_factor(int n, double* A, int* piv) {
    for (int i = 0; i < n; ++i) piv[i] = i;
    for (int k = 0; k < n; ++k) {
        int p = k;
        double maxv = std::fabs(A[k * n + k]);
        for (int i = k + 1; i < n; ++i) {
            double v = std::fabs(A[i * n + k]);
            if (v > maxv) {
                maxv = v;
                p = i;
            }
        }
        if (maxv < 1e-30 || !std::isfinite(maxv)) return false;
        if (p != k) {
            std::swap(piv[k], piv[p]);
            for (int j = 0; j < n; ++j) {
                std::swap(A[k * n + j], A[p * n + j]);
            }
        }
        const double inv_diag = 1.0 / A[k * n + k];
        for (int i = k + 1; i < n; ++i) {
            const double mult = A[i * n + k] * inv_diag;
            A[i * n + k] = mult;
            for (int j = k + 1; j < n; ++j) {
                A[i * n + j] -= mult * A[k * n + j];
            }
        }
    }
    return true;
}

inline void lu_solve(int n, const double* LU, const int* piv, const double* b, double* x) {
    for (int i = 0; i < n; ++i) {
        double sum = b[piv[i]];
        for (int j = 0; j < i; ++j) {
            sum -= LU[i * n + j] * x[j];
        }
        x[i] = sum;
    }
    for (int i = n - 1; i >= 0; --i) {
        double sum = x[i];
        for (int j = i + 1; j < n; ++j) {
            sum -= LU[i * n + j] * x[j];
        }
        x[i] = sum / LU[i * n + i];
    }
}

// Integrate single trajectory using compact active subsystem with adaptive L-stable ROS2.
int integrate_ros2_single(
    const NetworkDesc& net,
    const int* rxn_mask,
    const double* log10_k,
    const double* y0,
    const double* t_eval,
    int n_t,
    double rtol,
    double atol,
    int max_steps,
    double* y_out,
    double* flux_half1,
    double* flux_half2
) {
    CompactSubsys sub;
    if (!build_compact_subsys(net, rxn_mask, log10_k, sub)) return 1;

    const int ns_full = net.n_species;
    const int nr_full = net.n_reactions;
    const int ns = sub.n_as;
    const int nr = sub.n_ar;

    // Pre-fill constant inactive species across all output timestamps
    for (int ti = 0; ti < n_t; ++ti) {
        std::memcpy(y_out + ti * ns_full, y0, sizeof(double) * ns_full);
    }
    if (flux_half1) std::memset(flux_half1, 0, sizeof(double) * nr_full);
    if (flux_half2) std::memset(flux_half2, 0, sizeof(double) * nr_full);
    if (ns == 0 || nr == 0) return 0;

    double y[MAX_S], y_trial[MAX_S], y_stage[MAX_S];
    double f0[MAX_S], f1[MAX_S], f_next[MAX_S];
    double k1[MAX_S], k2[MAX_S], rhs2[MAX_S], err_raw[MAX_S], err_filt[MAX_S];
    double J[MAX_S * MAX_S], A[MAX_S * MAX_S];
    int piv[MAX_S];
    double r_curr[MAX_R], r_next[MAX_R];

    for (int s = 0; s < ns; ++s) y[s] = y0[sub.sub_to_full_s[s]];

    const double t_end = t_eval[n_t - 1];
    const double t_mid = 0.5 * t_end;
    double t = 0.0;
    int out_idx = 0;
    while (out_idx < n_t && t_eval[out_idx] <= 0.0) {
        ++out_idx;
    }
    if (out_idx >= n_t) return 0;

    const bool track_flux = (flux_half1 != nullptr || flux_half2 != nullptr);
    eval_rhs_jac_compact(sub, y, f0, J, track_flux ? r_curr : nullptr);

    double d0_sq = 0.0, d1_sq = 0.0;
    for (int s = 0; s < ns; ++s) {
        double sc = atol + rtol * std::fabs(y[s]);
        double v0 = y[s] / sc;
        double v1 = f0[s] / sc;
        d0_sq += v0 * v0;
        d1_sq += v1 * v1;
    }
    double d0 = std::sqrt(d0_sq / ns);
    double d1 = std::sqrt(d1_sq / ns);
    double h = (d0 < 1e-5 || d1 < 1e-5) ? 1e-6 : 0.01 * (d0 / d1);
    h = std::min(std::max(h, 1e-10), std::min(1e-2, t_end));

    int n_steps = 0;
    int n_rejects = 0;
    const double neg_tol = -std::max(1e-8, 10.0 * atol);

    while (t < t_end - 1e-15 && n_steps < max_steps) {
        if (t + h > t_end) h = t_end - t;
        if (h < 1e-15) return 2;

        const double gh = GAMMA_ROS2 * h;
        for (int i = 0; i < ns; ++i) {
            for (int j = 0; j < ns; ++j) {
                A[i * ns + j] = ((i == j) ? 1.0 : 0.0) - gh * J[i * ns + j];
            }
        }

        if (!lu_factor(ns, A, piv)) {
            h *= 0.25;
            ++n_rejects;
            if (n_rejects > max_steps) return 3;
            continue;
        }

        lu_solve(ns, A, piv, f0, k1);

        for (int s = 0; s < ns; ++s) y_stage[s] = y[s] + h * k1[s];
        eval_rhs_compact(sub, y_stage, f1, nullptr);
        for (int s = 0; s < ns; ++s) rhs2[s] = f1[s] - 2.0 * k1[s];
        lu_solve(ns, A, piv, rhs2, k2);

        bool finite_ok = true;
        bool pos_ok = true;
        for (int s = 0; s < ns; ++s) {
            double yn = y[s] + 1.5 * h * k1[s] + 0.5 * h * k2[s];
            if (!std::isfinite(yn)) {
                finite_ok = false;
                break;
            }
            if (yn < neg_tol) pos_ok = false;
            y_trial[s] = yn;
            err_raw[s] = 0.5 * h * (k1[s] + k2[s]);
        }

        if (!finite_ok || !pos_ok) {
            h *= 0.35;
            ++n_rejects;
            if (n_rejects > max_steps) return 4;
            continue;
        }

        lu_solve(ns, A, piv, err_raw, err_filt);

        double err_sq = 0.0;
        for (int s = 0; s < ns; ++s) {
            double sc = atol + rtol * std::max(std::fabs(y[s]), std::fabs(y_trial[s]));
            double e = err_filt[s] / sc;
            err_sq += e * e;
        }
        double err_norm = std::sqrt(err_sq / ns);

        if (err_norm <= 1.0) {
            const double t_new = t + h;
            ++n_steps;
            const bool is_last = (t_new >= t_end - 1e-15);

            if (is_last) {
                eval_rhs_compact(sub, y_trial, f_next, track_flux ? r_next : nullptr);
            } else {
                eval_rhs_jac_compact(sub, y_trial, f_next, J, track_flux ? r_next : nullptr);
            }

            if (track_flux) {
                double w1 = 0.0, w2 = 0.0;
                if (t_new <= t_mid) {
                    w1 = h;
                } else if (t >= t_mid) {
                    w2 = h;
                } else {
                    w1 = t_mid - t;
                    w2 = t_new - t_mid;
                }
                for (int r = 0; r < nr; ++r) {
                    double avg_r = 0.5 * (r_curr[r] + r_next[r]);
                    int fr = sub.ar_full_idx[r];
                    if (flux_half1 && w1 > 0.0) flux_half1[fr] += w1 * avg_r;
                    if (flux_half2 && w2 > 0.0) flux_half2[fr] += w2 * avg_r;
                }
            }

            while (out_idx < n_t && t_eval[out_idx] <= t_new + 1e-14) {
                double tau = (t_eval[out_idx] - t) / h;
                double* row = y_out + out_idx * ns_full;
                if (tau <= 0.0) {
                    for (int s = 0; s < ns; ++s) row[sub.sub_to_full_s[s]] = y[s];
                } else if (tau >= 1.0) {
                    for (int s = 0; s < ns; ++s) row[sub.sub_to_full_s[s]] = y_trial[s];
                } else {
                    double tau2 = tau * tau;
                    double tau3 = tau2 * tau;
                    double h00 = 2.0 * tau3 - 3.0 * tau2 + 1.0;
                    double h10 = (tau3 - 2.0 * tau2 + tau) * h;
                    double h01 = -2.0 * tau3 + 3.0 * tau2;
                    double h11 = (tau3 - tau2) * h;
                    bool hermite_ok = true;
                    double tmp[MAX_S];
                    for (int s = 0; s < ns; ++s) {
                        double val = h00 * y[s] + h10 * f0[s] + h01 * y_trial[s] + h11 * f_next[s];
                        if (val < neg_tol) {
                            hermite_ok = false;
                            break;
                        }
                        tmp[s] = val;
                    }
                    if (hermite_ok) {
                        for (int s = 0; s < ns; ++s) row[sub.sub_to_full_s[s]] = tmp[s];
                    } else {
                        for (int s = 0; s < ns; ++s) row[sub.sub_to_full_s[s]] = (1.0 - tau) * y[s] + tau * y_trial[s];
                    }
                }
                ++out_idx;
            }

            t = t_new;
            for (int s = 0; s < ns; ++s) {
                y[s] = y_trial[s];
                f0[s] = f_next[s];
            }
            if (track_flux) {
                for (int r = 0; r < nr; ++r) r_curr[r] = r_next[r];
            }

            double fac = (err_norm < 1e-12) ? 4.0 : 0.9 * std::pow(err_norm, -0.5);
            fac = std::min(std::max(fac, 0.2), 4.0);
            h = std::min(h * fac, 0.25 * t_end);
        } else {
            double fac = 0.85 * std::pow(err_norm, -0.5);
            fac = std::min(std::max(fac, 0.1), 0.5);
            h *= fac;
            ++n_rejects;
            if (n_rejects > max_steps) return 5;
        }
    }

    if (t < t_end - 1e-12) return 6;

    while (out_idx < n_t) {
        double* row = y_out + out_idx * ns_full;
        for (int s = 0; s < ns; ++s) row[sub.sub_to_full_s[s]] = y[s];
        ++out_idx;
    }
    return 0;
}

int eval_multi_design_residuals(
    const NetworkDesc& net,
    const int* rxn_mask,
    const double* active_theta,
    const int* active_slots,
    int n_active,
    int n_designs,
    const double* y0_all,
    const int* obs_flat,
    const int* obs_offsets,
    const double* t_eval,
    int n_t,
    const double* target_flat,
    const double* inv_sigma_flat,
    double rtol,
    double atol,
    int max_steps,
    double* res_out,
    double* y_buf
) {
    double log10_k[MAX_P];
    std::memset(log10_k, 0, sizeof(double) * net.n_params);
    for (int i = 0; i < n_active; ++i) {
        log10_k[active_slots[i]] = active_theta[i];
    }

    const int ns = net.n_species;
    int rpos = 0;
    for (int d = 0; d < n_designs; ++d) {
        int st = integrate_ros2_single(
            net, rxn_mask, log10_k,
            y0_all + d * ns, t_eval, n_t,
            rtol, atol, max_steps, y_buf, nullptr, nullptr
        );
        if (st != 0) return st;

        const int obeg = obs_offsets[d];
        const int oend = obs_offsets[d + 1];
        for (int oi = obeg; oi < oend; ++oi) {
            const int sp = obs_flat[oi];
            for (int ti = 0; ti < n_t; ++ti) {
                double pred = y_buf[ti * ns + sp];
                res_out[rpos] = (pred - target_flat[rpos]) * inv_sigma_flat[rpos];
                ++rpos;
            }
        }
    }
    return 0;
}

double fit_lm_single(
    const NetworkDesc& net,
    const int* rxn_mask,
    const int* active_slots,
    int n_active,
    const double* theta_init,
    const double* lb,
    const double* ub,
    int n_designs,
    const double* y0_all,
    const int* obs_flat,
    const int* obs_offsets,
    const double* t_eval,
    int n_t,
    const double* target_flat,
    const double* inv_sigma_flat,
    int n_res,
    double rtol,
    double atol,
    int max_ode_steps,
    int max_lm_iters,
    double stop_rms,
    double* theta_best_out
) {
    const int ns = net.n_species;
    std::vector<double> y_buf(n_t * ns);
    std::vector<double> res(n_res), res_pert(n_res), res_trial(n_res);
    std::vector<double> Jmat(n_res * n_active);

    double theta[MAX_P], theta_trial[MAX_P];
    for (int i = 0; i < n_active; ++i) {
        theta[i] = std::min(std::max(theta_init[i], lb[i]), ub[i]);
    }

    int st = eval_multi_design_residuals(
        net, rxn_mask, theta, active_slots, n_active,
        n_designs, y0_all, obs_flat, obs_offsets, t_eval, n_t,
        target_flat, inv_sigma_flat, rtol, atol, max_ode_steps,
        res.data(), y_buf.data()
    );
    if (st != 0) {
        for (int i = 0; i < n_active; ++i) theta_best_out[i] = theta[i];
        return 1e9;
    }

    double cost = 0.0;
    for (int i = 0; i < n_res; ++i) cost += res[i] * res[i];
    double best_cost = cost;
    for (int i = 0; i < n_active; ++i) theta_best_out[i] = theta[i];

    const double rms_cut = std::max(1e-3, stop_rms);
    if (std::sqrt(cost / n_res) <= rms_cut) {
        return std::sqrt(best_cost / n_res);
    }

    double lambda = 1e-2;
    double JTJ[MAX_P * MAX_P], A_lm[MAX_P * MAX_P];
    double JTr[MAX_P], delta[MAX_P];
    int piv[MAX_P];
    const double fd_step = std::max(1e-5, 2.0 * std::sqrt(rtol));

    for (int iter = 0; iter < max_lm_iters; ++iter) {
        if (std::sqrt(cost / n_res) <= rms_cut) break;

        bool jac_ok = true;
        for (int p = 0; p < n_active; ++p) {
            double orig = theta[p];
            double eps = fd_step;
            if (orig + eps > ub[p]) eps = -fd_step;
            theta[p] = orig + eps;
            int stp = eval_multi_design_residuals(
                net, rxn_mask, theta, active_slots, n_active,
                n_designs, y0_all, obs_flat, obs_offsets, t_eval, n_t,
                target_flat, inv_sigma_flat, rtol, atol, max_ode_steps,
                res_pert.data(), y_buf.data()
            );
            theta[p] = orig;
            if (stp != 0) {
                jac_ok = false;
                break;
            }
            const double inv_eps = 1.0 / eps;
            for (int i = 0; i < n_res; ++i) {
                Jmat[i * n_active + p] = (res_pert[i] - res[i]) * inv_eps;
            }
        }
        if (!jac_ok) break;

        std::memset(JTJ, 0, sizeof(double) * n_active * n_active);
        std::memset(JTr, 0, sizeof(double) * n_active);
        for (int i = 0; i < n_res; ++i) {
            const double ri = res[i];
            const double* Ji = &Jmat[i * n_active];
            for (int p = 0; p < n_active; ++p) {
                const double Jip = Ji[p];
                JTr[p] -= Jip * ri;
                for (int q = p; q < n_active; ++q) {
                    JTJ[p * n_active + q] += Jip * Ji[q];
                }
            }
        }
        for (int p = 0; p < n_active; ++p) {
            for (int q = 0; q < p; ++q) {
                JTJ[p * n_active + q] = JTJ[q * n_active + p];
            }
        }

        double gnorm = 0.0;
        for (int p = 0; p < n_active; ++p) gnorm = std::max(gnorm, std::fabs(JTr[p]));
        if (gnorm < 1e-8) break;

        bool accepted = false;
        for (int tries = 0; tries < 7; ++tries) {
            for (int p = 0; p < n_active; ++p) {
                for (int q = 0; q < n_active; ++q) {
                    A_lm[p * n_active + q] = JTJ[p * n_active + q];
                }
                double diag = JTJ[p * n_active + p];
                A_lm[p * n_active + p] += lambda * (diag > 1e-6 ? diag : 1e-6) + 1e-10;
            }

            if (!lu_factor(n_active, A_lm, piv)) {
                lambda *= 10.0;
                continue;
            }
            lu_solve(n_active, A_lm, piv, JTr, delta);

            double step_norm = 0.0;
            for (int p = 0; p < n_active; ++p) {
                double dp = std::min(std::max(delta[p], -1.5), 1.5);
                theta_trial[p] = std::min(std::max(theta[p] + dp, lb[p]), ub[p]);
                step_norm = std::max(step_norm, std::fabs(theta_trial[p] - theta[p]));
            }
            if (step_norm < 1e-7) {
                accepted = false;
                break;
            }

            int stt = eval_multi_design_residuals(
                net, rxn_mask, theta_trial, active_slots, n_active,
                n_designs, y0_all, obs_flat, obs_offsets, t_eval, n_t,
                target_flat, inv_sigma_flat, rtol, atol, max_ode_steps,
                res_trial.data(), y_buf.data()
            );
            if (stt == 0) {
                double trial_cost = 0.0;
                for (int i = 0; i < n_res; ++i) trial_cost += res_trial[i] * res_trial[i];
                if (trial_cost < cost) {
                    double rel_improv = (cost - trial_cost) / (cost + 1e-12);
                    cost = trial_cost;
                    res.swap(res_trial);
                    for (int p = 0; p < n_active; ++p) theta[p] = theta_trial[p];
                    if (cost < best_cost) {
                        best_cost = cost;
                        for (int p = 0; p < n_active; ++p) theta_best_out[p] = theta[p];
                    }
                    lambda = std::max(lambda * 0.3, 1e-9);
                    accepted = true;
                    if (rel_improv < 2e-4) {
                        return std::sqrt(best_cost / n_res);
                    }
                    break;
                }
            }
            lambda = std::min(lambda * 5.0, 1e8);
        }

        if (!accepted) break;
    }

    return std::sqrt(best_cost / n_res);
}

}  // anonymous namespace

extern "C" {

void simulate_batch_ros2(
    int batch,
    int n_species,
    int n_reactions,
    int n_params,
    const int* reactant1,
    const int* reactant2,
    const int* rxn_param_slot,
    const double* stoich_matrix,
    const int* rxn_masks,
    const double* log10_ks,
    const double* y0s,
    const double* t_eval,
    int n_t,
    double rtol,
    double atol,
    int max_steps,
    int n_threads,
    double* y_out,
    double* flux1_out,
    double* flux2_out,
    int* status_out
) {
    NetworkDesc net;
    build_network_desc(
        n_species, n_reactions, n_params,
        reactant1, reactant2, rxn_param_slot, stoich_matrix, net
    );

    if (n_threads > 0) omp_set_num_threads(n_threads);

    #pragma omp parallel for schedule(dynamic, 32)
    for (int i = 0; i < batch; ++i) {
        double* f1 = flux1_out ? (flux1_out + static_cast<size_t>(i) * n_reactions) : nullptr;
        double* f2 = flux2_out ? (flux2_out + static_cast<size_t>(i) * n_reactions) : nullptr;
        status_out[i] = integrate_ros2_single(
            net,
            rxn_masks + static_cast<size_t>(i) * n_reactions,
            log10_ks + static_cast<size_t>(i) * n_params,
            y0s + static_cast<size_t>(i) * n_species,
            t_eval,
            n_t,
            rtol,
            atol,
            max_steps,
            y_out + static_cast<size_t>(i) * n_t * n_species,
            f1,
            f2
        );
    }
}

void fit_lm_batch(
    int n_tasks,
    int n_starts,
    int n_species,
    int n_reactions,
    int n_params,
    const int* reactant1,
    const int* reactant2,
    const int* rxn_param_slot,
    const double* stoich_matrix,
    const int* mech_rxn_masks,
    const int* mech_param_masks,
    const int* task_mech_ids,
    int n_designs,
    const double* mech_y0_all,
    const int* obs_flat,
    const int* obs_offsets,
    const double* t_eval,
    int n_t,
    int n_res,
    const double* targets_flat,
    const double* inv_sigmas_flat,
    const double* init_thetas,
    const double* lb_all,
    const double* ub_all,
    double rtol,
    double atol,
    int max_ode_steps,
    int max_lm_iters,
    double early_stop_rms,
    int n_threads,
    double* best_rms_out,
    double* best_theta_out
) {
    NetworkDesc net;
    build_network_desc(
        n_species, n_reactions, n_params,
        reactant1, reactant2, rxn_param_slot, stoich_matrix, net
    );

    if (n_threads > 0) omp_set_num_threads(n_threads);

    #pragma omp parallel for schedule(dynamic, 4)
    for (int t = 0; t < n_tasks; ++t) {
        const int mid = task_mech_ids[t];
        const int* rmask = mech_rxn_masks + static_cast<size_t>(mid) * n_reactions;
        const int* pmask = mech_param_masks + static_cast<size_t>(mid) * n_params;
        const double* y0_m = mech_y0_all + static_cast<size_t>(mid) * n_designs * n_species;
        const double* lb_m = lb_all + static_cast<size_t>(mid) * n_params;
        const double* ub_m = ub_all + static_cast<size_t>(mid) * n_params;

        int active_slots[MAX_P];
        double lb_a[MAX_P], ub_a[MAX_P], init_a[MAX_P], out_a[MAX_P];
        int n_active = 0;
        for (int p = 0; p < n_params; ++p) {
            if (pmask[p]) {
                active_slots[n_active] = p;
                lb_a[n_active] = lb_m[p];
                ub_a[n_active] = ub_m[p];
                ++n_active;
            }
        }

        const double* target_t = targets_flat + static_cast<size_t>(t) * n_res;
        const double* invs_t = inv_sigmas_flat + static_cast<size_t>(t) * n_res;
        double* best_theta_t = best_theta_out + static_cast<size_t>(t) * n_params;
        std::memset(best_theta_t, 0, sizeof(double) * n_params);

        double best_rms = 1e9;
        for (int s = 0; s < n_starts; ++s) {
            const double* init_full = init_thetas + (static_cast<size_t>(t) * n_starts + s) * n_params;
            for (int k = 0; k < n_active; ++k) {
                init_a[k] = init_full[active_slots[k]];
            }
            double rms = fit_lm_single(
                net, rmask, active_slots, n_active,
                init_a, lb_a, ub_a,
                n_designs, y0_m, obs_flat, obs_offsets, t_eval, n_t,
                target_t, invs_t, n_res,
                rtol, atol, max_ode_steps, max_lm_iters, early_stop_rms, out_a
            );
            if (rms < best_rms) {
                best_rms = rms;
                for (int k = 0; k < n_active; ++k) {
                    best_theta_t[active_slots[k]] = out_a[k];
                }
            }
            if (early_stop_rms > 0.0 && best_rms <= early_stop_rms) {
                break;
            }
        }
        best_rms_out[t] = best_rms;
    }
}

// Pair-grouped directional multi-start LM with EXACT binomial early exit:
// For each pair p in [0, n_pairs), evaluates K samples sequentially in a thread;
// if more than max_fails samples exceed fail_rms_thresh (making it mathematically
// impossible to meet the pass-rate threshold), prunes remaining samples of pair p
// (setting rms = 999.0) with ZERO approximation to the equivalence test.
void fit_lm_pairs_dir(
    int n_pairs,
    int K,
    int n_starts,
    int n_species,
    int n_reactions,
    int n_params,
    const int* reactant1,
    const int* reactant2,
    const int* rxn_param_slot,
    const double* stoich_matrix,
    const int* mech_rxn_masks,
    const int* mech_param_masks,
    const int* pair_target_mech_ids,  // [n_pairs]: mechanism being fitted
    int n_designs,
    const double* mech_y0_all,
    const int* obs_flat,
    const int* obs_offsets,
    const double* t_eval,
    int n_t,
    int n_res,
    const double* targets_flat,       // [n_pairs * K * n_res]
    const double* inv_sigma_1d,       // [n_res] shared across all tasks
    const double* init_thetas,        // [n_pairs * K * n_starts * n_params]
    const double* lb_all,
    const double* ub_all,
    double rtol,
    double atol,
    int max_ode_steps,
    int max_lm_iters,
    double early_stop_rms,
    double fail_rms_thresh,
    int max_fails,                    // break pair early when n_fails > max_fails (<0 disables)
    int n_threads,
    double* best_rms_out              // [n_pairs * K]
) {
    NetworkDesc net;
    build_network_desc(
        n_species, n_reactions, n_params,
        reactant1, reactant2, rxn_param_slot, stoich_matrix, net
    );

    if (n_threads > 0) omp_set_num_threads(n_threads);

    std::atomic<int> done_cnt{0};
    const int report_step = std::max(1, n_pairs / 5);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int p = 0; p < n_pairs; ++p) {
        const int mid = pair_target_mech_ids[p];
        const int* rmask = mech_rxn_masks + static_cast<size_t>(mid) * n_reactions;
        const int* pmask = mech_param_masks + static_cast<size_t>(mid) * n_params;
        const double* y0_m = mech_y0_all + static_cast<size_t>(mid) * n_designs * n_species;
        const double* lb_m = lb_all + static_cast<size_t>(mid) * n_params;
        const double* ub_m = ub_all + static_cast<size_t>(mid) * n_params;

        int active_slots[MAX_P];
        double lb_a[MAX_P], ub_a[MAX_P], init_a[MAX_P], out_a[MAX_P];
        int n_active = 0;
        for (int q = 0; q < n_params; ++q) {
            if (pmask[q]) {
                active_slots[n_active] = q;
                lb_a[n_active] = lb_m[q];
                ub_a[n_active] = ub_m[q];
                ++n_active;
            }
        }

        int n_fails = 0;
        bool pruned = false;
        for (int k = 0; k < K; ++k) {
            const size_t task_idx = static_cast<size_t>(p) * K + k;
            if (pruned) {
                best_rms_out[task_idx] = 999.0;
                continue;
            }
            const double* target_t = targets_flat + task_idx * n_res;
            double best_rms = 1e9;
            for (int s = 0; s < n_starts; ++s) {
                const double* init_full = init_thetas + (task_idx * n_starts + s) * n_params;
                for (int j = 0; j < n_active; ++j) {
                    init_a[j] = init_full[active_slots[j]];
                }
                double rms = fit_lm_single(
                    net, rmask, active_slots, n_active,
                    init_a, lb_a, ub_a,
                    n_designs, y0_m, obs_flat, obs_offsets, t_eval, n_t,
                    target_t, inv_sigma_1d, n_res,
                    rtol, atol, max_ode_steps, max_lm_iters, early_stop_rms, out_a
                );
                if (rms < best_rms) best_rms = rms;
                if (early_stop_rms > 0.0 && best_rms <= early_stop_rms) break;
            }
            best_rms_out[task_idx] = best_rms;
            if (max_fails >= 0 && best_rms > fail_rms_thresh) {
                ++n_fails;
                if (n_fails > max_fails) pruned = true;
            }
        }
        int c = ++done_cnt;
        if (n_pairs >= 200 && (c % report_step == 0 || c == n_pairs)) {
            std::printf("    [LM-pairs] %d / %d pairs done\n", c, n_pairs);
            std::fflush(stdout);
        }
    }
}

}  // extern "C"
