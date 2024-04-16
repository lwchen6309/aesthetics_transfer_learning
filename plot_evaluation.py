import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def eval_PARA(model=None):
    dfs = []
    for i_cv in range(1,5):
        if model is not None:
            filename = 'PARA_%s_evaluation_results%d.csv'%(model, i_cv)
        else:
            filename = 'PARA_evaluation_results%d.csv'%i_cv
        df = pd.read_csv(filename)
        dfs.append(df)
    df = pd.concat(dfs)
    scopes = ['PIAA_Score', 'age', 'gender', 'education', 'art_experience', 'photo_experience']
    # Generate results
    model_label = 'NIMA' if model is None else model
    results = generate_results(df, scopes, model_label)
    return results


def eval_LAPIS(model=None):
    plt.close()
    dfs = []
    for i_cv in range(1,5):
        if model is not None:
            filename = 'LAPIS_%s_evaluation_results%d.csv'%(model, i_cv)
        else:
            filename = 'LAPIS_evaluation_results%d.csv'%i_cv
        df = pd.read_csv(filename)
        dfs.append(df)
    df = pd.concat(dfs)
    scopes = ['PIAA_Score', 'age', 'demo_gender', 'demo_edu', 'nationality']
    model_label = 'NIMA' if model is None else model
    results = generate_results(df, scopes, model_label)
    return results

def generate_results(df, scopes, model_label='NIMA', loss_field='EMD_Loss_Data'):
    results = []
    for scope in scopes:
        true_scores = df[scope]
        unique_scores, mean_emd_loss, std_emd_loss = [], [], []
        for score, group in df.groupby(scope):
            unique_scores.append(score)
            mean_emd_loss.append(group[loss_field].mean())
            std_emd_loss.append(group[loss_field].std())
        
        results.append({
            'scope': scope,
            'model_label': model_label,            
            'unique_scores': unique_scores,
            'category_counts': {score: sum(true_scores == score) for score in unique_scores},
            'mean_emd_loss': np.array(mean_emd_loss),
            'std_emd_loss': np.array(std_emd_loss)
        })
    return results


def plot_results_bak(results_list, dataname):
    # Create a color map or predefined color list for visual distinction between models
    colors = ['skyblue', 'salmon', 'lightgreen', 'gold']
    markers = ['o', 's', 'v', '^']
    
    for scope_index, scope_results in enumerate(zip(*results_list)):  # Unzip by scope
        plt.figure(figsize=(10, 12))
        
        # Assume all entries in results_list have the same scopes in the same order
        scope = scope_results[0]['scope']  # scope name from the first result in scope_results
        
        # First subplot for the histogram of true_scores
        plt.subplot(2, 1, 1)
        for i, result in enumerate(scope_results):
            unique_scores = result['unique_scores']
            category_counts = result['category_counts']
            plt.bar(np.array(range(len(unique_scores))) + i*0.1, list(category_counts.values()), width=0.1, color=colors[i % len(colors)], edgecolor='black', label=result.get('model_label', 'Base Model'))
        plt.xticks(ticks=np.arange(len(unique_scores)), labels=unique_scores)
        plt.title(f'Histogram of {scope}', fontsize=16)
        plt.xlabel(scope, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()

        # Second subplot for the mean and std of EMD Loss Data
        plt.subplot(2, 1, 2)
        for i, result in enumerate(scope_results):
            unique_scores = result['unique_scores']
            mean_emd_loss = result['mean_emd_loss']
            std_emd_loss = result['std_emd_loss']
            plt.errorbar(unique_scores, mean_emd_loss, yerr=std_emd_loss, fmt=markers[i % len(markers)], ecolor='red', capsize=5, capthick=2, label=result.get('model_label', 'Base Model'))
        plt.xlabel(scope, fontsize=12)
        plt.ylabel('EMD Loss Data', fontsize=12)
        plt.title(f'{scope} vs EMD Loss Data (Mean and STD)', fontsize=16)
        plt.legend()

        plt.tight_layout()
        figname = f"{dataname}_comparison_{scope}.png"
        plt.savefig(figname)
        plt.close()


def plot_results(results_list, dataname):
    colors = ['skyblue', 'salmon', 'lightgreen', 'gold']
    markers = ['o', 's', 'v', '^']

    for scope_index, scope_results in enumerate(zip(*results_list)):  # Unzip by scope
        plt.figure(figsize=(10, 12))
        
        scope = scope_results[0]['scope']
        
        # Detect if unique_scores are numeric or categorical
        first_result = scope_results[0]
        if all(isinstance(score, (int, float, np.number)) for score in first_result['unique_scores']):
            is_numeric = True
        else:
            is_numeric = False
            # Create a mapping from categorical labels to numeric indices
            categories = {score: idx for idx, score in enumerate(set(score for result in scope_results for score in result['unique_scores']))}

        # First subplot for the histogram
        plt.subplot(2, 1, 1)
        for i, result in enumerate(scope_results):
            unique_scores = result['unique_scores']
            category_counts = result['category_counts']
            if is_numeric:
                bar_positions = np.array(unique_scores) + 0.1 * i
            else:
                bar_positions = [categories[score] + 0.1 * i for score in unique_scores]

            plt.bar(bar_positions, list(category_counts.values()), width=0.1, color=colors[i % len(colors)], edgecolor='black', label=result.get('model_label', 'Base Model'))

        plt.xticks(ticks=[categories[score] + 0.1 * (len(results_list) - 1) / 2 if not is_numeric else score + 0.1 * (len(results_list) - 1) / 2 for score in unique_scores], labels=unique_scores)
        plt.title(f'Histogram of {scope}', fontsize=16)
        plt.xlabel(scope, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()

        # Second subplot for the EMD Loss Data
        plt.subplot(2, 1, 2)
        for i, result in enumerate(scope_results):
            unique_scores = result['unique_scores']
            mean_emd_loss = result['mean_emd_loss']
            std_emd_loss = result['std_emd_loss']
            if is_numeric:
                error_bar_positions = np.array(unique_scores) + 0.1 * i
            else:
                error_bar_positions = [categories[score] + 0.1 * i for score in unique_scores]

            plt.errorbar(error_bar_positions, mean_emd_loss, yerr=std_emd_loss, fmt=markers[i % len(markers)], color=colors[i % len(colors)], ecolor='red', capsize=5, capthick=2, label=result.get('model_label', 'Base Model'))
        
        plt.xlabel(scope, fontsize=12)
        plt.ylabel('EMD Loss Data', fontsize=12)
        plt.title(f'{scope} vs EMD Loss Data (Mean and STD)', fontsize=16)
        plt.legend()

        plt.tight_layout()
        figname = f"{dataname}_comparison_{scope}.png"
        plt.savefig(figname)
        plt.close()



if __name__ == '__main__':
    para_nima_results = eval_PARA()
    para_lf_results = eval_PARA('LF')
    para_lf_IS_results = eval_PARA('LF_IS')
    para_results = [para_nima_results, para_lf_results, para_lf_IS_results]
    plot_results(para_results, dataname='PARA')
    
    # lapis_nima_results = eval_LAPIS()
    # plot_results(lapis_nima_results, dataname='LAPIS')