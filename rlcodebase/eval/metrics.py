import abc

import numpy as np
from scipy.stats import ttest_ind, spearmanr

from rlcodebase.infrastructure.utils.rl_utils import convert_listofrollouts
from rlcodebase.eval.behavior_policy import BehaviorPolicy


class EvalMetricBase(abc.ABC):
    def __init__(self, name):
        self.name = name
        self.requires_fitting = False

    @abc.abstractmethod
    def value(self, ob_no, av_ac_n, ac_na, re_n, terminal_n, rtgs_n, q_vals_n, opt_ac_na):
        pass

    @abc.abstractmethod
    def fit(self, train_paths, test_paths, params, eval_policy):
        pass


class WIS(EvalMetricBase):
    def __init__(self, eps):
        super().__init__('WIS')
        self.requires_fitting = True

        self.eps = eps

        self.ac_dim = None
        self.behavior_policy = BehaviorPolicy()

    def fit(self, train_paths, test_paths, params, eval_policy):
        ob_no, opt_ac_n, _, _, _, _ = convert_listofrollouts(train_paths)

        self.ac_dim = np.max(opt_ac_n) + 1

        print('fitting behavior policy')
        self.behavior_policy.update(ob_no, opt_ac_n)
        print('done!')

        self.requires_fitting = False

    def rho(self, ob, ac, opt_ac):
        """
        @param ob: observation
        @param ac: policy action
        @param opt_ac: behavior policy action
        @return:
        """
        p_b = self.behavior_policy.get_action_probs(ob[np.newaxis, :], opt_ac[np.newaxis])[0]
        p = (1 - self.eps) if ac == opt_ac else self.eps/(self.ac_dim - 1)
        return p/p_b

    def value(self, ob_no, av_ac_n, ac_na, re_n, terminal_n, rtgs_n, q_vals_n, opt_ac_na):
        n = len(terminal_n)

        all_rtgs = []
        all_rhos = []
        all_rho_prods = []

        # Find rtg and rho_prod
        rtg = rtgs_n[0]
        rhos = []
        for t in range(n):
            rho = self.rho(ob_no[t], ac_na[t], opt_ac_na[t])
            rhos.append(rho)
            if terminal_n[t] == 1:
                all_rtgs.append(rtg)
                all_rhos.append(rhos)
                all_rho_prods.append(np.prod(rhos))

                rtg = rtgs_n[(t + 1) % n]
                rhos = []

        # Find w
        longest = np.max([len(rhos) for rhos in all_rhos])
        w = np.ones((longest,))
        for t in range(longest):
            w[t] = np.mean([np.prod(rhos[: (t + 1)]) for rhos in all_rhos])

        # Find WIS
        n_traj = len(all_rtgs)
        vals = np.zeros((n_traj,))
        for i_traj in range(n_traj):
            len_traj = len(all_rhos[i_traj])
            vals[i_traj] = all_rho_prods[i_traj] * all_rtgs[i_traj] / w[len_traj - 1]

        return {
            'mean': np.mean(vals),
            'std': np.std(vals),
            'n': n_traj
        }


class CurrentPolicyReturn(EvalMetricBase):
    def __init__(self):
        super().__init__('Current_Policy_Return')

    def fit(self, train_paths, test_paths, params, eval_policy):
        pass

    def value(self, ob_no, av_ac_n, ac_na, re_n, terminal_n, rtgs_n, q_vals_n, opt_ac_na):
        n = len(terminal_n)

        all_rtgs = []

        # Find rtg and rho_prod
        rtg = rtgs_n[0]
        for t in range(n):
            if terminal_n[t] == 1:
                all_rtgs.append(rtg)

                rtg = rtgs_n[(t + 1) % n]

        n_traj = len(all_rtgs)
        return {
            'mean': np.mean(all_rtgs),
            'std': np.std(all_rtgs),
            'n': n_traj
        }


class TTest(EvalMetricBase):
    def __init__(self, th=0):
        super().__init__('T_Stat')

        self.th = th

    def value(self, ob_no, av_ac_n, ac_na, re_n, terminal_n, rtgs_n, q_vals_n, opt_ac_na):
        res = ttest_ind(
            q_vals_n[rtgs_n > self.th],
            q_vals_n[rtgs_n <= self.th],
            equal_var=False,
            alternative='greater'
        )
        # noinspection PyTypeChecker, PyUnresolvedReferences
        return {
            'statistic': res.statistic,
            'pvalue': res.pvalue
        }

    def fit(self, train_paths, test_paths, params, eval_policy):
        pass


class Accuracy(EvalMetricBase):
    def __init__(self):
        super().__init__('Accuracy')

    def value(self, ob_no, av_ac_n, ac_na, re_n, terminal_n, rtgs_n, q_vals_n, opt_ac_na):
        if not isinstance(ac_na, np.ndarray):
            ac_na = np.ndarray(ac_na)
        if not isinstance(opt_ac_na, np.ndarray):
            opt_ac_na = np.array(opt_ac_na)

        p = np.mean(ac_na == opt_ac_na)
        n = len(ac_na)
        return {
            'mean': p,
            'std': p*(1 - p),
            'n': n
        }

    def fit(self, train_paths, test_paths, params, eval_policy):
        pass


class Recall(EvalMetricBase):
    def __init__(self):
        super().__init__('Recall')

    def value(self, ob_no, av_ac_n, ac_na, re_n, terminal_n, rtgs_n, q_vals_n, opt_ac_na):
        p = np.mean([(opt_ac in av_ac_n[i_ac]) for i_ac, opt_ac in enumerate(opt_ac_na)])

        n = len(ac_na)
        return {
            'mean': p,
            'std': p*(1 - p),
            'n': n
        }

    def fit(self, train_paths, test_paths, params, eval_policy):
        pass


class NumAvailable(EvalMetricBase):
    def __init__(self):
        super().__init__('Num_of_Available_Actions')

    def value(self, ob_no, av_ac_n, ac_na, re_n, terminal_n, rtgs_n, q_vals_n, opt_ac_na):
        n_av_ac = [len(av_ac) for av_ac in av_ac_n]

        n = len(n_av_ac)
        return {
            'mean': np.mean(n_av_ac),
            'std': np.std(n_av_ac),
            'n': n
        }

    def fit(self, train_paths, test_paths, params, eval_policy):
        pass


class RhoBinned(EvalMetricBase):
    def __init__(self, bins):
        super().__init__('Rho_Binned')

        self.bins = bins

    def value(self, ob_no, av_ac_n, ac_na, re_n, terminal_n, rtgs_n, q_vals_n, opt_ac_na):
        qs = np.linspace(0, 1, self.bins + 1)

        quantiles = np.quantile(q_vals_n, qs)

        avg_rtg_bins = np.zeros((self.bins,))
        for i_bin in range(self.bins):
            l = quantiles[i_bin]
            r = quantiles[i_bin + 1]

            bin_filt = np.logical_and(q_vals_n >= l, q_vals_n < r)

            avg_rtg_bins[i_bin] = np.mean(rtgs_n[bin_filt])

        # noinspection PyTypeChecker, PyUnresolvedReferences
        corr = spearmanr(range(self.bins), avg_rtg_bins).correlation

        return {
            'mean': corr,
            'std': 0.01,
            'n': 1
        }

    def fit(self, train_paths, test_paths, params, eval_policy):
        pass


class DiffSurvivalQuantiles(EvalMetricBase):
    def __init__(self, q, th):
        super().__init__('Diff_Survival_Quantiles')

        self.q = q
        self.th = th

    def value(self, ob_no, av_ac_n, ac_na, re_n, terminal_n, rtgs_n, q_vals_n, opt_ac_na):
        qs = [0, self.q, 1 - self.q, 1]

        quantiles = np.quantile(q_vals_n, qs)

        avg_rtg_bins = np.zeros((3,))
        for i_bin in range(3):
            l = quantiles[i_bin]
            r = quantiles[i_bin + 1]

            bin_filt = np.logical_and(q_vals_n >= l, q_vals_n < r)

            avg_rtg_bins[i_bin] = np.mean(rtgs_n[bin_filt] > self.th)

        p_1 = avg_rtg_bins[0]
        p_3 = avg_rtg_bins[-1]

        n = len(rtgs_n)
        return {
            'mean': p_3 - p_1,
            'std': p_3*(1 - p_3) + p_1*(1 - p_1),
            'n': n
        }

    def fit(self, train_paths, test_paths, params, eval_policy):
        pass


def get_default_metrics():
    return [Accuracy(), CurrentPolicyReturn()]


def get_default_offline_metrics():
    return [
        Accuracy(), CurrentPolicyReturn(), RhoBinned(bins=100), DiffSurvivalQuantiles(q=0.25, th=0), TTest(th=0),
        WIS(eps=0.01), FQE(n_iter=80001)
    ]


def get_default_pruning_metrics():
    return [Recall(), NumAvailable()]
