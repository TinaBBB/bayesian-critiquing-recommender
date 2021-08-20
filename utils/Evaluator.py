import numpy as np


class Evaluator:
    def __init__(self, rec_atK):
        self.rec_atK = rec_atK  # a list of the topK indecies
        self.rec_maxK = max(self.rec_atK)

        self.global_metrics = {
            "R-Precision": r_precision,
            "NDCG": ndcg
        }

        self.local_metrics = {
            "Precision": precisionk,
            "Recall": recallk,
            "MAP": average_precisionk,
            "NDCG": ndcg
        }

    def evaluate(self, model, dataset, test_batch_size, mse_only, ndcg_only,
                 analytical=False):
        model.eval()
        model.before_evaluate()

        # Get MSE value
        preds, ys = model.predict(dataset, test_batch_size)
        RMSE = np.sqrt((np.sum((preds - ys) ** 2)) / len(ys))

        if mse_only:
            recommendation_results = {"RMSE": RMSE}
            return recommendation_results
        else:
            pred_matrix = model.simple_predict(dataset.train_matrix)
            assert pred_matrix.shape == dataset.train_matrix.shape

            # get predicted item index
            prediction = []
            num_users = pred_matrix.shape[0]

            # Prediction section
            for user_index in range(num_users):
                vector_prediction = pred_matrix[user_index]
                vector_train = dataset.train_matrix[user_index]

                if len(vector_train.nonzero()[0]) > 0:
                    vector_predict = sub_routine(vector_prediction, vector_train, topK=self.rec_maxK)
                else:
                    vector_predict = np.zeros(self.rec_maxK, dtype=np.float32)

                prediction.append(vector_predict)

            predicted_items = prediction.copy()

            recommendation_results = self.evaluation(predicted_items, dataset.test_matrix, eval_type='recommendations',
                                                     ndcg_only=ndcg_only, analytical=analytical)
            recommendation_results["RMSE"] = RMSE
        return recommendation_results

    def evaluate_recommendations(self, model, input_matrix, test_matrix, ndcg_only,
                                 analytical=False):
        # switch to evaluation mode
        model.eval()
        # operations before evaluation, does not perform for VAE models
        model.before_evaluate()

        # get prediction data, in matrix form
        pred_matrix = model.predict(input_matrix)
        assert pred_matrix.shape == input_matrix.shape

        # get predicted item index
        prediction = []

        num_users = pred_matrix.shape[0]

        # Prediction section
        for user_index in range(num_users):
            vector_prediction = pred_matrix[user_index]
            vector_train = input_matrix[user_index]
            # test_index_length = len(test_matrix[user_index].nonzero()[1])
            # topK = max(test_index_length, self.rec_maxK)  # max of number of ground truth entries and topK

            if len(vector_train.nonzero()[0]) > 0:
                vector_predict = sub_routine(vector_prediction, vector_train, topK=self.rec_maxK)
            else:
                vector_predict = np.zeros(self.rec_maxK, dtype=np.float32)

            prediction.append(vector_predict)

        # predicted item indecies
        # predicted_items = np.vstack(prediction)
        predicted_items = prediction.copy()

        recommendation_results = self.evaluation(predicted_items, test_matrix, eval_type='recommendations',
                                                 ndcg_only=ndcg_only, analytical=analytical)

        return recommendation_results

     # function to perform evaluation on metrics
    def evaluation(self, predicted_items, test_matrix, eval_type, ndcg_only, analytical=False):
        if eval_type == 'recommendations' and ndcg_only:
            local_metrics = None
            global_metrics = {"NDCG": ndcg}
            atK = self.rec_atK
        elif eval_type == 'recommendations' and not ndcg_only:
            local_metrics = self.local_metrics
            global_metrics = self.global_metrics
            atK = self.rec_atK
        elif eval_type == 'embeddings' and ndcg_only:
            local_metrics = None
            global_metrics = {"UK_NDCG": ndcg}
            atK = self.explain_atK
        elif eval_type == 'embeddings' and not ndcg_only:
            local_metrics = self.local_metrics_embeddings
            global_metrics = self.global_metrics_embeddings
            atK = self.explain_atK
        else:
            raise NotImplementedError("Please select proper evaluation type, current choice: %s" % eval_type)

        num_users = test_matrix.shape[0]

        # evaluation section
        output = dict()

        # The @K metrics
        if local_metrics:
            for k in atK:
                results = {name: [] for name in local_metrics.keys()}

                # topK_Predict = predicted_items[:, :k]
                for user_index in range(num_users):
                    # vector_predict = topK_Predict[user_index]
                    vector_predict = predicted_items[user_index][:k]
                    if (len(vector_predict.nonzero()[0]) > 0):
                        vector_true_dense = test_matrix[user_index].nonzero()[1]

                        if vector_true_dense.size > 0:  # only if length of validation set is not 0
                            hits = np.isin(vector_predict, vector_true_dense)
                            for name in local_metrics.keys():
                                results[name].append(local_metrics[name](vector_true_dense=vector_true_dense,
                                                                         vector_predict=vector_predict,
                                                                         hits=hits))

                results_summary = dict()
                if analytical:
                    for name in local_metrics.keys():
                        results_summary['{0}@{1}'.format(name, k)] = results[name]
                else:
                    for name in local_metrics.keys():
                        results_summary['{0}@{1}'.format(name, k)] = (np.average(results[name]),
                                                                      1.96 * np.std(results[name]) / np.sqrt(
                                                                          len(results[name])))
                output.update(results_summary)

        # The global metrics
        results = {name: [] for name in global_metrics.keys()}
        for user_index in range(num_users):
            vector_predict = predicted_items[user_index]

            if len(vector_predict.nonzero()[0]) > 0:
                vector_true_dense = test_matrix[user_index].nonzero()[1]
                hits = np.isin(vector_predict, vector_true_dense)

                if vector_true_dense.size > 0:
                    for name in global_metrics.keys():
                        results[name].append(global_metrics[name](vector_true_dense=vector_true_dense,
                                                                  vector_predict=vector_predict, hits=hits))
        results_summary = dict()
        if analytical:
            for name in global_metrics.keys():
                results_summary[name] = results[name]
        else:
            for name in global_metrics.keys():
                results_summary[name] = (
                    np.average(results[name]), 1.96 * np.std(results[name]) / np.sqrt(len(results[name])))
        output.update(results_summary)

        return output


def sub_routine(vector_predict, vector_train, topK):
    train_index = vector_train.nonzero()[1]

    # take the top recommended items
    candidate_index = np.argpartition(-vector_predict, topK + len(train_index))[:topK + len(train_index)]
    vector_predict = candidate_index[vector_predict[candidate_index].argsort()[::-1]]

    # vector_predict = np.argsort(-vector_predict)[:topK + len(train_index)]
    vector_predict = np.delete(vector_predict, np.isin(vector_predict, train_index).nonzero()[0])

    return vector_predict[:topK]


# ranking metrics for recommendation performance
def recallk(vector_true_dense, hits, **unused):
    hits = len(hits.nonzero()[0])
    return float(hits) / len(vector_true_dense)


def precisionk(vector_predict, hits, **unused):
    hits = len(hits.nonzero()[0])
    return float(hits) / len(vector_predict)


def average_precisionk(vector_predict, hits, **unused):
    precisions = np.cumsum(hits, dtype=np.float32) / range(1, len(vector_predict) + 1)
    return np.mean(precisions)


def r_precision(vector_true_dense, vector_predict, **unused):
    vector_predict_short = vector_predict[:len(vector_true_dense)]
    hits = len(np.isin(vector_predict_short, vector_true_dense).nonzero()[0])
    return float(hits) / len(vector_true_dense)


def _dcg_support(size):
    arr = np.arange(1, size + 1) + 1
    return 1. / np.log2(arr)


def ndcg(vector_true_dense, vector_predict, hits):
    idcg = np.sum(_dcg_support(len(vector_true_dense)))
    dcg_base = _dcg_support(len(vector_predict))
    dcg_base[np.logical_not(hits)] = 0
    dcg = np.sum(dcg_base)
    return dcg / idcg
