import torch
import scipy
from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from typing import AnyStr, List
from slingpy import AbstractDataSource
from slingpy.models.abstract_base_model import AbstractBaseModel
from genedisco.active_learning_methods.acquisition_functions.base_acquisition_function import \
    BaseBatchAcquisitionFunction


class CoreSet(object):
    def __call__(self, dataset_x: AbstractDataSource, batch_size: int, available_indices: List[AnyStr],
                 last_selected_indices: List[AnyStr], last_model: AbstractBaseModel) -> List:
        topmost_hidden_representation = last_model.get_embedding(dataset_x.subset(available_indices)).numpy()
        selected_hidden_representations = last_model.get_embedding(dataset_x.subset(last_selected_indices)).numpy()
        chosen = self.select_most_distant(topmost_hidden_representation, selected_hidden_representations, batch_size)
        return [available_indices[idx] for idx in chosen]

    def select_most_distant(self, options, previously_selected, num_samples):
        num_options, num_selected = len(options), len(previously_selected)
        if num_selected == 0:
            min_dist = np.tile(float("inf"), num_options)
        else:
            dist_ctr = pairwise_distances(options, previously_selected)
            min_dist = np.amin(dist_ctr, axis=1)

        indices = []
        for i in range(num_samples):
            idx = min_dist.argmax()
            dist_new_ctr = pairwise_distances(options, options[[idx], :])
            for j in range(num_options):
                min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])
            indices.append(idx)
        return indices

class AdversarialBIM(object):
    def __init__(self, args=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if args is None:
            args = {}

        if 'eps' in args:
            self.eps = args['eps']
        else:
            self.eps = 0.05

        if 'verbose' in args:
            self.verbose = args['verbose']
        else:
            self.verbose = True

        if 'stop_iterations_by_count' in args:
            self.stop_iterations_by_count = args['stop_iterations_by_count']
        else:
            self.stop_iterations_by_count = 1000

        if 'gamma' in args:
            self.gamma = args['gamma']
        else:
            self.gamma = 0.35

        if 'adversarial_sample_ratio' in args:
            self.adversarial_sample_ratio = args['adversarial_sample_ratio']
        else:
            self.adversarial_sample_ratio = 0.1

        super(AdversarialBIM, self).__init__()

    def __call__(self, dataset_x: AbstractDataSource, batch_size: int, available_indices: List[AnyStr],
                 last_selected_indices: List[AnyStr], last_model: AbstractBaseModel) -> List:
        dis = np.zeros(len(available_indices)) + np.inf
        data_pool = dataset_x.subset(available_indices)

        for i, index in enumerate(available_indices[:100]):
            x = torch.as_tensor(data_pool.subset([index]).get_data()).to(self.device)
            dis[i] = self.cal_dis(x, last_model)

        chosen = dis.argsort()[:batch_size]
        for x in np.sort(dis)[:batch_size]:
            print(x)
        return [available_indices[idx] for idx in chosen]

    def cal_dis(self, x, last_model):
        nx = x.detach()
        first_x = torch.clone(nx)

        nx.requires_grad_()
        eta = torch.zeros(nx.shape).to(self.device)
        iteration = 0

        while torch.linalg.norm(nx + eta - first_x) < self.gamma * torch.linalg.norm(first_x):

            if iteration >= self.stop_iterations_by_count:
                break

            out = torch.as_tensor(last_model.get_model_prediction(nx + eta, return_multiple_preds=True)[0])
            out = torch.squeeze(out)
            variance = torch.var(out)
            variance.backward()

            eta += self.eps * torch.sign(nx.grad.data)
            nx.grad.data.zero_()
            iteration += 1
        return (eta * eta).sum()


class TopUncertainAcquisition(object):
    def __call__(self,
                 dataset_x: AbstractDataSource,
                 select_size: int,
                 available_indices: List[AnyStr],
                 last_selected_indices: List[AnyStr] = None,
                 model: AbstractBaseModel = None,
                 ) -> List:
        avail_dataset_x = dataset_x.subset(available_indices)
        model_pedictions = model.predict(avail_dataset_x, return_std_and_margin=True)

        if len(model_pedictions) != 3:
            raise TypeError("The provided model does not output uncertainty.")

        pred_mean, pred_uncertainties, _ = model_pedictions

        if len(pred_mean) < select_size:
            raise ValueError("The number of query samples exceeds"
                             "the size of the available data.")

        numerical_selected_indices = np.flip(
            np.argsort(pred_uncertainties)
        )[:select_size]
        selected_indices = [available_indices[i] for i in numerical_selected_indices]

        return selected_indices


class SoftUncertainAcquisition(object):
    def __call__(self,
                 dataset_x: AbstractDataSource,
                 select_size: int,
                 available_indices: List[AnyStr],
                 last_selected_indices: List[AnyStr] = None,
                 model: AbstractBaseModel = None,
                 temperature: float = 0.9,
                 ) -> List:
        avail_dataset_x = dataset_x.subset(available_indices)
        model_pedictions = model.predict(avail_dataset_x, return_std_and_margin=True)

        if len(model_pedictions) != 3:
            raise TypeError("The provided model does not output uncertainty.")

        pred_mean, pred_uncertainties, _ = model_pedictions

        if len(pred_mean) < select_size:
            raise ValueError("The number of query samples exceeds"
                             "the size of the available data.")
        selection_probabilities = self.softmax_temperature(
            np.log(1e-7 + pred_uncertainties ** 2),
            temperature,
        )
        numerical_selected_indices = np.random.choice(
            range(len(selection_probabilities)),
            size=select_size,
            replace=False,
            p=selection_probabilities)
        selected_indices = [available_indices[i] for i
                            in numerical_selected_indices]
        return selected_indices

    def softmax_temperature(self, x, temperature=1):
        """Computes softmax probabilities from unnormalized values

        Args:

            x: array-like list of energy values.
            temperature: a positive real value.

        Returns:
            outputs: ndarray or list (dependin on x type) that is
                exp(x / temperature) / sum(exp(x / temperature)).
        """
        if isinstance(x, list):
            y = np.array(x)
        else:
            y = x
        y = np.exp(y / temperature)
        out_np = scipy.special.softmax(y)
        if any(np.isnan(out_np)):
            raise ValueError("Temperature is too extreme.")
        if isinstance(x, list):
            return [out_item for out_item in out_np]
        else:
            return out_np


class RandomBatchAcquisitionFunction(object):
    def __call__(self, dataset_x, batch_size, available_indices, last_selected_indices, last_model):
        selected = np.random.choice(available_indices, size=batch_size, replace=False)
        return selected


class KmeansBatchAcquisitionFunction(object):
    def __init__(self, representation="linear", n_init=10):
        """
            is embedding: Apply kmeans to embedding or raw data
            n_init: Specifies the number of kmeans run-throughs to use, wherein the one with the smallest inertia is
                selected for the selection phase
        """
        self.representation = representation
        self.n_init = n_init
        super(KmeansBatchAcquisitionFunction, self).__init__()

    def __call__(self, dataset_x: AbstractDataSource, batch_size: int, available_indices: List[AnyStr],
                 last_selected_indices: List[AnyStr], last_model: AbstractBaseModel) -> List:
        if self.representation == 'linear':
            kmeans_dataset = last_model.get_embedding(dataset_x.subset(available_indices)).numpy()
        elif self.representation == 'raw':
            kmeans_dataset = np.squeeze(dataset_x.subset(available_indices), axis=1)
        else:
            raise ValueError("Representation must be one of 'linear', 'raw'")

        centers = self.kmeans_clustering(kmeans_dataset, batch_size)
        chosen = self.select_closest_to_centers(kmeans_dataset, centers)
        return [available_indices[idx] for idx in chosen]

    def kmeans_clustering(self, kmeans_dataset, num_centers):
        kmeans = KMeans(init='k-means++', n_init=self.n_init, n_clusters=num_centers).fit(kmeans_dataset)
        return kmeans.cluster_centers_

    def select_closest_to_centers(self, options, centers):
        dist_ctr = pairwise_distances(options, centers)
        min_dist_indices = np.argmin(dist_ctr, axis=0)

        return list(min_dist_indices)


class BadgeSampling(object):
    def __call__(self, dataset_x: AbstractDataSource, batch_size: int, available_indices: List[AnyStr],
                 last_selected_indices: List[AnyStr], last_model: AbstractBaseModel) -> List:
        gradient_embedding = last_model.get_gradient_embedding(dataset_x.subset(available_indices)).numpy()
        chosen = BadgeSampling.kmeans_initialise(gradient_embedding, batch_size)
        selected = [available_indices[idx] for idx in chosen]
        return selected

    @staticmethod
    def kmeans_initialise(gradient_embedding, k):
        ind = np.argmax([np.linalg.norm(s, 2) for s in gradient_embedding])
        mu = [gradient_embedding[ind]]
        indsAll = [ind]
        centInds = [0.] * len(gradient_embedding)
        cent = 0
        while len(mu) < k:
            if len(mu) == 1:
                D2 = pairwise_distances(gradient_embedding, mu).ravel().astype(float)
            else:
                newD = pairwise_distances(gradient_embedding, [mu[-1]]).ravel().astype(float)
                for i in range(len(gradient_embedding)):
                    if D2[i] > newD[i]:
                        centInds[i] = cent
                        D2[i] = newD[i]
            D2 = D2.ravel().astype(float)
            Ddist = (D2 ** 2) / sum(D2 ** 2)
            customDist = scipy.stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
            ind = customDist.rvs(size=1)[0]
            mu.append(gradient_embedding[ind])
            indsAll.append(ind)
            cent += 1
        gram = np.matmul(gradient_embedding[indsAll], gradient_embedding[indsAll].T)
        val, _ = np.linalg.eig(gram)
        val = np.abs(val)
        vgt = val[val > 1e-2]
        return indsAll


class MarginSamplingAcquisition(object):
    def __call__(self,
                 dataset_x: AbstractDataSource,
                 select_size: int,
                 available_indices: List[AnyStr],
                 last_selected_indices: List[AnyStr] = None,
                 model: AbstractBaseModel = None,
                 ) -> List:
        avail_dataset_x = dataset_x.subset(available_indices)
        model_pedictions = model.predict(avail_dataset_x, return_std_and_margin=True)

        if len(model_pedictions) != 3:
            raise TypeError("The provided model does not output margins.")

        # print("Hi! This is Margin Sampling!")
        pred_mean, pred_uncertainties, pred_margins = model_pedictions
        # print(pred_margins)

        if len(pred_mean) < select_size:
            raise ValueError("The number of query samples exceeds"
                             "the size of the available data.")

        numerical_selected_indices = np.flip(
            np.argsort(pred_margins)
        )[:select_size]
        selected_indices = [available_indices[i] for i in numerical_selected_indices]

        return selected_indices


class BanditRewardsLog:
    def __init__(self):
        self.total_actions = 0
        self.total_rewards = 0
        self.all_rewards = []
        self.record = defaultdict(lambda: dict(actions=0, reward=0, reward_squared=0))

    def record_action(self, bandit_id, reward):
        self.total_actions += 1
        self.total_rewards += reward
        self.all_rewards.append(reward)
        self.record[bandit_id]['actions'] += 1
        self.record[bandit_id]['reward'] += reward
        self.record[bandit_id]['reward_squared'] += reward ** 2

    def __getitem__(self, bandit_id):
        return self.record[bandit_id]


class BanditAcquisition(BaseBatchAcquisitionFunction):
    def __init__(self, bandit_set="hitratio"):
        """
        :param bandit_set:"hitratio" or "mse"
        """
        if bandit_set == "hitratio":
            self.bandits = [CoreSet(), AdversarialBIM(), MarginSamplingAcquisition()]
        elif bandit_set == "mse":
            self.bandits = [KmeansBatchAcquisitionFunction(representation="raw"), TopUncertainAcquisition(),
                            MarginSamplingAcquisition(), RandomBatchAcquisitionFunction()]

        self.bandit_count = len(self.bandits)
        self.best_bandit_ratio = 0.9  # Originally in UBC1-Tuned this equals 1.0, but I've decided to provide an epsilon

        # initial ratios (uniform)
        self.bandit_ratios =[]
        for bandit_id in range(self.bandit_count):
            self.bandit_ratios.append(1.0 / self.bandit_count)

        self.rewards_log = BanditRewardsLog()
        self.last_select_bandit_ids = None
        self.last_select_bandit_mapping = None
        self.iteration = 0

    def __call__(self,
                 dataset_x: AbstractDataSource,
                 select_size: int,
                 available_indices: List[AnyStr], 
                 last_selected_indices: List[AnyStr] = None,
                 last_model: AbstractBaseModel = None
                 ) -> List:
        if self.iteration > 0:
            self._update_ratios(dataset_x, last_selected_indices, last_model)

        selected = self._select_according_to_ratios(dataset_x, select_size, available_indices, last_selected_indices,
                                                    last_model)
        self.iteration += 1

        return selected


    def print_ratios(self):
        print("=======================")
        for bandit_id in range(self.bandit_count):
            print(f"Bandit {bandit_id}={str(self.bandits[bandit_id])}, ratio: {self.bandit_ratios[bandit_id]}")
        print("=======================")

    def _update_ratios(self, dataset_x, last_selected_indices, last_model):
        previous_x = dataset_x.subset(last_selected_indices)

        try:
            model_pedictions = last_model.get_model_prediction(previous_x, False)
            y_pred = model_pedictions[0].reshape(-1).tolist()
        except:
            y_pred = last_model.predict(previous_x, return_std_and_margin=False)[0]

        for s_pos in range(len(last_selected_indices)):
            bandit_id = self.last_select_bandit_mapping[last_selected_indices[s_pos]]
            reward = abs(y_pred[s_pos])
            self.rewards_log.record_action(bandit_id, reward)

        best_bandit_id = self._get_current_best_bandit_id()

        for bandit_id in range(self.bandit_count):
            if bandit_id == best_bandit_id:
                self.bandit_ratios[bandit_id] = self.best_bandit_ratio
            else:
                self.bandit_ratios[bandit_id] = (1.0 - self.best_bandit_ratio) / (self.bandit_count - 1.0)

    def _select_according_to_ratios(self, dataset_x, select_size, available_indices, last_selected_indices, last_model):
        bandit_id = 0
        threshold = self.bandit_ratios[bandit_id] * select_size
        selection = []
        self.last_select_bandit_ids = []
        self.last_select_bandit_mapping = {}
        bandit_available_indices = available_indices
        self.print_ratios()

        for i in range(select_size):
            if i >= threshold and bandit_id < self.bandit_count - 1:
                bandit_id += 1
                threshold += self.bandit_ratios[bandit_id] * select_size
            self.last_select_bandit_ids.append(bandit_id)

        self.last_select_bandit_ids = np.array(self.last_select_bandit_ids)

        for bandit_id in range(self.bandit_count):
            badit_select_size = np.sum(self.last_select_bandit_ids == bandit_id)
            bandit_selection = self.bandits[bandit_id](dataset_x, badit_select_size, bandit_available_indices,
                                                       last_selected_indices, last_model)
            selection.extend(bandit_selection)
            bandit_available_indices = [i for i in bandit_available_indices if i not in bandit_selection]

            for s in bandit_selection:
                self.last_select_bandit_mapping[s] = bandit_id

        return selection

    def _get_current_best_bandit_id(self):
        estimates = [self._calculate_bandit_index(bandit_id) for bandit_id in range(self.bandit_count)]
        return np.argmax(estimates)

    def _calculate_bandit_index(self, bandit_id):
        """
        UBC1-Tuned estimate
        """
        bandit_record = self.rewards_log[bandit_id]
        n = bandit_record['actions']
        sample_mean = bandit_record['reward'] / n

        variance_bound = bandit_record['reward_squared'] / n - sample_mean ** 2
        variance_bound += np.sqrt(2 * np.log(self.rewards_log.total_actions) / n)

        c = np.sqrt(np.min([variance_bound, 1 / 4]) * np.log(self.rewards_log.total_actions) / n)
        return sample_mean + c