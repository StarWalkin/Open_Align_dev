import os

from codes.utils.utils import *
from numpyro.infer import MCMC, NUTS
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.util import set_platform
from matplotlib import pyplot as plt
import xgboost
import shap
from functools import partial
import sklearn
import itertools

# set plt canvas size
plt.rcParams["figure.figsize"] = (20, 10)

numpyro.set_host_device_count(4)


def extract_prefix(model_name):
    """
    Extract the prefix from the model name.
    """
    return model_name.split('-')[0]


def show_all_else_equal_prob(weights, model_name=None, suffix=""):
    # weights should be a 1D array
    posterior_means = np.asarray(weights)

    # Create a matrix where each row has one feature set to +1 and the rest are 0
    X_test = np.eye(weights.shape[0])
    if set == -1:
        X_test = -X_test

    # Calculate the logits by multiplying the test matrix with the posterior means
    logits = X_test @ posterior_means

    # Apply the sigmoid function to the logits to get the probabilities
    probabilities = 1 / (1 + np.exp(-logits))

    # Plot these probabilities
    plt.figure(figsize=(12, 6))
    plt.plot(range(0, weights.shape[0]), probabilities, marker='o')
    plt.title(f'{model_name} - Probability of +1 Label for an one-Hot feature (+1)')
    plt.xlabel('Feature Names')
    plt.ylabel('Probability of +1 Label')

    plt.xticks(range(0, weights.shape[0]), feature_names, rotation=90)
    plt.grid(True)
    plt.ylim(0.2, 0.8)

    # 在y=0.5处添加红色虚线
    plt.axhline(y=0.5, color='red', linestyle='--')

    # 调整边距
    plt.subplots_adjust(bottom=0.3)
    if not os.path.exists(f"./collected_data/model_preference/imgs_{suffix}"):
        os.mkdir(f"./collected_data/model_preference/imgs_{suffix}")
    plt.savefig(f"./collected_data/model_preference/imgs_{suffix}/{model_name}_probabilities.png")
    plt.show()


def show_all_else_equal_prob_forlarge(weights, model_name=None, ax=None, line_style='-'):
    """
    Plot the probabilities for a single set of weights.
    """
    # weights should be a 1D array
    posterior_means = weights

    # Create a matrix where each row has one feature set to +1 and the rest are 0
    X_test = np.eye(weights.shape[0])
    if set == -1:
        X_test = -X_test

    # Calculate the logits by multiplying the test matrix with the posterior means
    logits = X_test @ posterior_means

    # Apply the sigmoid function to the logits to get the probabilities
    probabilities = 1 / (1 + np.exp(-logits))

    # Plot these probabilities on the provided axes
    ax.plot(range(0, weights.shape[0]), probabilities, marker='o', linestyle=line_style, label=model_name)


def get_label(item, type="human_preference", which_model=None, which_model_preference=None):
    if type == "human_preference":
        label_str = item['source']['winner']
    elif type == "model_preference":
        assert which_model is not None
        assert which_model_preference is not None
        label_str = item['source']['model_preference'][which_model][which_model_preference]
    else:
        raise NotImplementedError

    if label_str == "model_a":
        return 1
    elif label_str == "model_b":
        return 0
    else:
        raise ValueError


def model_preference_as_label(file, type="qrlogprob"):
    if type == "qrlogprob":
        data = read_all(file)
        labels = []
        for item in data:
            if item["response_a"] > item["response_b"]:
                xx = 1
            else:
                xx = 0
            labels.append(xx)
        return np.asarray(labels)


def bayesian_logistic_regression(X, y, scale=0.01):
    # Priors for the regression coefficients
    alpha = numpyro.sample('alpha', dist.Laplace(loc=jnp.zeros(X.shape[1]), scale=scale))

    # Calculate the linear predictor (the logits) using JAX NumPy
    logits = jnp.dot(X, alpha)

    # Likelihood of the observations given the logistic model
    with numpyro.plate('data', X.shape[0]):
        numpyro.sample('obs', dist.Bernoulli(logits=logits), obs=y)


def repeat_data(X, y, repeat=1):
    # repeat X and y on the first axis to get more samples
    X = np.repeat(X, repeat, axis=0)
    y = np.repeat(y, repeat, axis=0)
    return X, y


class BayesianLogisticRegression:
    def __init__(self, alpha):
        self.alpha = alpha

    def predict(self, X, thres=0.5):
        probs = self.return_prob(X)
        # Applying the threshold to make predictions
        # predictions = (probs >= thres).astype(int)
        predictions = np.round(probs)
        return predictions

    def return_prob(self, X):
        logits = np.dot(X, self.alpha)
        # return probabilities
        return np.exp(logits) / (1 + np.exp(logits))

class VanillaLogisticRegression:
    def __init__(self,model):
        self.model = model

    def predict(self,X, thres=0.5):
        # probs = self.return_prob(X)
        # predictions = (probs >= thres).astype(int)
        predictions = self.model.predict(X)
        return predictions

    def return_prob(self,X):
        return self.model.predict_proba(X)[:, 1]

def fit_bayes_logistic_regression(X, y, model_name, show=True, scale=0.01, suffix=""):
    # repeat X and y on the first axis to get more samples

    bxx = partial(bayesian_logistic_regression, scale=scale)

    kernel = NUTS(bxx)
    mcmc = MCMC(kernel, num_warmup=500, num_samples=2000, num_chains=4, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(0), X, y)
    # mcmc.print_summary()
    # raise ValueError
    # Get the posterior samples
    posterior_samples = mcmc.get_samples()

    # Compute the mean of the posterior for each alpha_i
    alpha_mean = np.mean(posterior_samples['alpha'], axis=0).tolist()

    if show:
        show_all_else_equal_prob(alpha_mean, model_name=model_name, suffix=suffix)

    return BayesianLogisticRegression(alpha_mean), alpha_mean


def fit_logisticregression(X, y, model_name, show=True, penalty=None, add_bias=False, C=1.0, ):
    if penalty == 'l1':
        model = sklearn.linear_model.LogisticRegression(fit_intercept=add_bias, penalty=penalty, solver='liblinear', C=C,max_iter=10000)
    else:
        model = sklearn.linear_model.LogisticRegression(fit_intercept=add_bias, penalty=penalty, C=C, max_iter=10000)
    model.fit(X, y)

    if show:
        show_all_else_equal_prob(model.coef_[0], model_name=model_name)

    return VanillaLogisticRegression(model), model.coef_[0].tolist()


def fit_XGBClassifier(X, y, repeat=1):
    X, y = repeat_data(X, y, repeat=repeat)
    model = xgboost.XGBRegressor()
    model.fit(X, y)
    return model


def fit_simple_MLP(X, y, repeat=1):
    from sklearn.neural_network import MLPClassifier
    X, y = repeat_data(X, y, repeat=repeat)
    model = MLPClassifier()
    model.fit(X, y)
    return model


def draw_all(all_parameters):
    """
    Plot the probabilities for all sets of parameters in the same figure, with same prefix models having same color but different line styles.
    """
    # Create a figure for the plots
    fig, ax = plt.subplots(figsize=(20, 12))

    # Initialize color and line style management
    color_map = {}
    line_styles = itertools.cycle(['-', '--', '-.', ':'])  # Define more line styles if needed
    colors = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])  # Add more colors if needed

    # Iterate over all sets of parameters
    for param in all_parameters:
        model_name = param['model_name']
        prefix = extract_prefix(model_name)
        weights = param['parameters']

        # Assign color and line style based on prefix
        if True:
            color_map[prefix] = next(colors)
        line_style = next(line_styles)

        # Set the color for plotting
        plt.gca().set_prop_cycle(color=[color_map[prefix]])

        show_all_else_equal_prob_forlarge(weights, model_name, ax, line_style)

    # Set labels and title
    ax.set_title('Probability of +1 Label for One-Hot Features Across Models')
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Probability of +1 Label')
    ax.set_xticks(range(0, weights.shape[0]), feature_names, rotation=90)
    ax.grid(True)
    ax.legend()

    # Display the plot
    plt.show()


def fit_regression(X, y, method="bayesian", preference_from="human", show=False, scale=0.1, penalty=None,
                   add_bias=False, C=1.0, suffix=""):
    # print_colored_text(f"Fitting Regression on {len(X)} samples ...")
    if method == "bayesian":
        model, parameters = fit_bayes_logistic_regression(X, y, model_name=preference_from, show=show, scale=scale,
                                                          suffix=suffix)
    elif method == "vanilla":
        model, parameters = fit_logisticregression(X, y, model_name=preference_from, show=show, penalty=penalty,
                                                   add_bias=add_bias, C=C)
    else:
        raise NotImplementedError
    return model, parameters


feature_names = list(feature_name_to_id_short.keys())

havemapping = {"human": [""],
               # "gpt-4-1106-preview": ["inst_mt_bench", "inst_mt_bench_by-exchange", "inst_naive_openai", "inst_naive_openai_by-exchange"],
               # "gpt-3.5-turbo-1106": ["inst_naive_openai", "inst_naive_openai_by-exchange"]
               }

if __name__ == '__main__':
    resolved_data = read_all("./collected_data/v1.4.0_resolved/chatbot_arena_no-tie_group_balanced_resolved.jsonl")
    query_aware_idxs = read_json("./collected_data/v1.4.0_resolved/query_aware_idxs.json")
    label_data = read_all("./collected_data/model_preference/labels/chatbot_arena_shuffled_no-tie_group_balanced_labels.jsonl")

    if_shap = False
    if_k_fold_then_avg = True
    get_cls_acc = False
    individual_show = False
    all_show = False

    comp_diff = "comparison"

    fitting_method = "bayesian"
    fitting_hyperpara = 0.1
    fitting_penalty = None

    key_list = list(query_aware_idxs.keys())

    accs = {"all":[0,0],"sep":[0,0]}

    for k in key_list:
    # for k in ['all']:
        all_xx = read_all(f"./collected_data/model_preference_finegrained/fitted_paras_{comp_diff}/model_{k}_fitted_paras.jsonl")

        print_colored_text(f"Query-aware idxs: {k}", color="cyan")
        cared_idxs = query_aware_idxs.get(k, list(range(20000)))
        for model_name in ["human"] + os.listdir("./collected_data/model_preference"):
        # for model_name in ['gpt-4-1106-preview']:
            listxx = havemapping.get(model_name, ["inst_naive_by-exchange"])
            for further_divide in listxx:
                inter = False
                print_colored_text(f"Model: {model_name} | Further divide: {further_divide}", color="green")

                features = []
                labels = []
                feature_count = {k: 0 for k in feature_name_to_id.keys()}

                for idx, item in enumerate(resolved_data):
                    if idx not in cared_idxs: continue
                    if item['comparison']['accuracy']['comparison'] == 999: continue
                    try:
                        # pprint(label_data[idx])
                        # raise ValueError
                        label = label_data[idx][model_name]
                        if further_divide != "": label = label[further_divide]
                    except:
                        print_colored_text(f"Fail to find {model_name} - {further_divide} for idx {idx}", color="red")
                        inter = True
                        break
                    feature = get_feature(item, remove_length=False, way=comp_diff)
                    features.append(feature)
                    labels.append(label)

                    for idxx in range(len(feature)):
                        if feature[idxx] != 0:
                            feature_count[feature_id_to_name[idxx]] += 1

                # print(feature_count)
                # raise ValueError

                if inter: continue

                features = np.asarray(features, dtype=np.float32)
                labels = np.asarray(labels)

                # random shuffle both features and labels, and keep the correspondence
                # set seed
                np.random.seed(0)
                idxs = np.arange(len(features))
                np.random.shuffle(idxs)
                features = features[idxs]
                labels = labels[idxs]

                # count the ratio of 1/0 in labels
                num_of_1 = np.count_nonzero(labels)
                print_colored_text(f"A is preferred: {num_of_1} | B is preferred: {len(labels) - num_of_1}")

                if get_cls_acc:
                    assert if_k_fold_then_avg == False
                    # take 0.8/0.2 train/test split
                    features_len = len(features)
                    split_point = int(0.8 * features_len)
                    features_train, features_test = features[:split_point, :], features[split_point:, :]
                    labels_train, labels_test = labels[:split_point], labels[split_point:]
                else:
                    features_train, features_test = features, None
                    labels_train, labels_test = labels, None

                if not if_k_fold_then_avg:
                    model, parameters = fit_regression(features_train, labels_train, method=fitting_method,
                                                   preference_from=f"{model_name} [{further_divide}]",
                                                   show=individual_show,
                                                   scale=fitting_hyperpara, penalty=fitting_penalty, C=fitting_hyperpara, suffix=k)
                else:
                    final_paras = None
                    for i in range(10):
                        # take the i/10 as test set
                        features_len = len(features)
                        split_point = int(i / 10 * features_len)
                        features_train, features_test = np.concatenate([features[:split_point, :], features[split_point + int(features_len / 10):, :]], axis=0), features[split_point:split_point + int(features_len / 10), :]
                        labels_train, labels_test = np.concatenate([labels[:split_point], labels[split_point + int(features_len / 10):]], axis=0), labels[split_point:split_point + int(features_len / 10)]
                        model, parameters = fit_regression(features_train, labels_train, method=fitting_method,
                                                           preference_from=f"{model_name} [{further_divide}]",
                                                           show=individual_show,
                                                           scale=fitting_hyperpara, penalty=fitting_penalty,
                                                           C=fitting_hyperpara, suffix=k)
                        if final_paras is None:
                            final_paras = np.asarray(parameters)
                        else:
                            final_paras += np.asarray(parameters)
                        print_colored_text(f"Fold {i} done!", color="green")
                    final_paras /= 10
                    parameters = final_paras.tolist()


                if get_cls_acc:
                    assert not if_k_fold_then_avg
                    predictions = model.predict(features_test)
                    accuracy = sklearn.metrics.accuracy_score(labels_test, predictions)
                    print(f"{model_name} >>> accuracy:", accuracy)
                    if k=="all":
                        accs["all"] = [accuracy*len(labels_test),len(labels_test)]
                    elif k not in ['intent_unclear','express_feeling','w_constraints','w_mistakes','w_stances']:
                        accs["sep"][0]+=accuracy*len(labels_test)
                        accs["sep"][1]+=len(labels_test)
                if if_shap:
                    explainer = shap.Explainer(model=model.predict, masker=np.zeros((1,features.shape[1])))
                    shap_values = explainer(features)

                    shap_values.feature_names = feature_names

                    shap.summary_plot(shap_values, plot_size=[16, 10], max_display=30, show=False)

                    plt.title(model_name)
                    # safe save with dir creating
                    if not os.path.exists(f"./collected_data/model_preference/imgs_{k}"):
                        os.mkdir(f"./collected_data/model_preference/imgs_{k}")
                    plt.savefig(f"./collected_data/model_preference/imgs_{k}/{model_name}_shap.png")
                    plt.show()
                    # raise ValueError

                all_xx.append({"model_name": f"{model_name} [{further_divide}]", "parameters": parameters})

                print(model_name, " ---- done!")

        if all_show:
            draw_all(all_xx)


        write_jsonl(all_xx, f"./collected_data/model_preference_finegrained/fitted_paras_{comp_diff}/model_{k}_fitted_paras.jsonl", mode="w")

    # print(accs,accs['all'][0]/accs['all'][1],accs['sep'][0]/accs['sep'][1])
