import pandas as pd
import matplotlib.pyplot as plt


def _draw_hyper_param_evolution_build_data(grid_search_results, hyper_param, greater_is_better):
    # Get the indexes of the current hyper param - with the other hyper params being constant
    grid_search_results_df = pd.DataFrame(grid_search_results.cv_results_)
    best_grid_item = grid_search_results_df[grid_search_results_df['rank_test_score'] == 1]['params'].values[0].copy()
    del best_grid_item[hyper_param]
    grid_search_results_df['best_other_params'] = grid_search_results_df['params'].apply(
        lambda y: all([y[key] == best_grid_item[key] for key in best_grid_item]))
    results_df = grid_search_results_df[grid_search_results_df['best_other_params'] == 1].copy()

    for sample, style, plotly_style in (('train', '--', 'dash'), ('test', '-', 'solid')):
        try:
            results_df.loc[:, 'mean_spread_low_%s' % sample] = \
                results_df['mean_%s_score' % sample] - results_df['std_%s_score' % sample]
            results_df['mean_spread_high_%s' % sample] = \
                results_df['mean_%s_score' % sample] + results_df['std_%s_score' % sample]

            results_df.sort_values(by=['param_' + hyper_param], inplace=True)

        except ValueError:
            print("INFO: Hyperparam {0} cannot be displayed. It is not a quantitative param".format(hyper_param))

    if greater_is_better:
        best_param = results_df['param_' + hyper_param].loc[results_df['mean_test_score'].idxmax(axis=0)]
        best_score = results_df['mean_test_score'].max()
    else:
        best_param = results_df['param_' + hyper_param].loc[results_df['mean_test_score'].idxmin(axis=0)]
        best_score = results_df['mean_test_score'].min()

    score = 'score'
    try:
        score = grid_search_results.estimator.criterion
    except:
        pass

    return results_df, best_score, best_param, score


def _draw_hyper_param_evolution(grid_search_results, hyperparam, greater_is_better):
    results_df, best_score, best_param, score = _draw_hyper_param_evolution_build_data(
        grid_search_results, hyperparam, greater_is_better)
    fig, ax = plt.subplots(1, 1)

    # plot:
    color = 'g'
    for sample, style in (('train', '--'), ('test', '-')):
        ax.fill_between(list(results_df['param_' + hyperparam].values),
                        list(results_df['mean_spread_low_%s' % sample].values),
                        list(results_df['mean_spread_high_%s' % sample].values),
                        alpha=0.1 if sample == 'test' else 0, color=color)

        ax.plot(list(results_df['param_' + hyperparam].values),
                list(results_df['mean_%s_score' % sample].values),
                style, color=color, alpha=1 if sample == 'test' else 0.7, label="score (%s)" % sample)

    # dotted vertical line at the best score for that scorer marked by x and annotate the best score for that scorer
    ax.plot([best_param], [best_score], linestyle='None', color=color, marker='x', markeredgewidth=3, ms=8)
    ax.plot([best_param, best_param], [0, best_score], color=color, linestyle='-.')
    ax.annotate("%0.2f" % best_score, (best_param, best_score + 0.005))
    ax.legend(loc="best")
    ax.grid(False)

    # annotate the figure:
    ax.set_title('Hyperparam_evolution', fontdict={'fontsize': 8})
    ax.set_xlabel(hyperparam)
    ax.set_ylabel(score)
    return fig, ax


def score_evolution_hyper_params(grid_search_results, hyper_params, greater_is_better, search_type):
    """
    Evolution of the scoring according to the hyper parameters values
    :param grid_search_results: results from the grid_search
    :param hyper_params: dictionary of the hyper_params passed to the grid search
    :param greater_is_better: whether low score is better
    :param search_type: whether grid or randomized. If the search is randomized, the visualization won't be drawn
    :return: Plots
    """
    if search_type == 'grid':
        figs = []
        keys = [k for k in hyper_params.keys() if len(hyper_params[k]) > 1 and not isinstance(hyper_params[k][0], str)
                and not isinstance(hyper_params[k][0], tuple)]
        if len(keys) > 0:
            for index, hyperparam in enumerate(keys):
                figs.append(_draw_hyper_param_evolution(grid_search_results, hyperparam, greater_is_better))
            return [fig[0] for fig in figs]
    else:
        print("INFO: No grid search has been run. "
              "Please, configure a grid search to analyse the hyper_params evolution")
        return None


def visualization_methods(method):
    """
    This function shows all the visualization that are available:
        - score_evolution_hyper_params
    """
    return {
        'score_evolution_hyper_params': score_evolution_hyper_params
    }.get(method)
