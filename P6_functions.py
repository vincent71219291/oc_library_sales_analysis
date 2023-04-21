import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import seaborn as sns
# from IPython.display import display

import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scikit_posthocs as sp


def describe_table(df, pk=None, return_res=False):
    if isinstance(pk, str):
        pk = [pk]
    result = [None, None]
    
    print(f"La table contient {df.shape[0]} observations et {df.shape[1]}",
          "variables.")
    print()
    
    print("Aperçu de la table")
    display(df.head(3))
    print("")
    
    print("Description des variables")
    var_desc = pd.DataFrame()
    var_desc['Compte'] = df.count()
    var_desc['Doublons'] = df.count() - df.nunique()
    var_desc['Valeurs manquantes'] = df.isna().sum()
    var_desc['Modalités'] = df.nunique()
    var_desc['Type'] = df.dtypes
    
    display(var_desc)
    print("")
    
    numerical_vars = df.select_dtypes('number')
    if not numerical_vars.empty:
        print("Description des variables numériques")
        numerical_var_desc = pd.DataFrame()
        numerical_var_desc['Min'] = numerical_vars.min()
        numerical_var_desc['Max'] = numerical_vars.max()
        numerical_var_desc['Moyenne'] = numerical_vars.mean()
        numerical_var_desc['Médiane'] = numerical_vars.median()
        display(numerical_var_desc)
        print("")
    
    # on vérifie que toutes les valeurs de la clé primaire sont uniques et
    # qu'aucune d'elles n'est manquante
    if pk:
        print("Clé primaire")
        print("------------")
        
        # doublons
        pk_dups = df.loc[df.duplicated(subset=pk, keep=False)]
        n_pk_dups = pk_dups.shape[0]
        if n_pk_dups:
            print(f"{n_pk_dups} valeurs de la clé primaire {pk} ne sont pas",
                  "uniques.")
        else:
            print(f"Toutes les valeurs de la clé primaire {pk} sont uniques.")
        if not pk_dups.empty:
            result[0] = pk_dups
        
        # valeurs manquantes
        n_nas = df[pk].isna().any(axis=1).sum()
        if n_nas:
            print(f"{n_nas} valeurs de {pk} sont manquantes.")
        else:
            print(f"Aucune des valeurs de {pk} n'est manquante.")
        
        pk_nas = df.loc[df[pk].isna().any(axis=1)]
        if not pk_nas.empty:
            result[1] = pk_nas

        if return_res:
            return result
        

def annotate_barplot(ax):
    y1_max_bboxes = 0
    plt.draw()

    for patch in ax.patches:
        x_min, y_min = patch.get_xy()
        x_max, y_max = x_min + patch.get_width(), y_min + patch.get_height()
        x_center = np.mean((x_min, x_max))
        annot = ax.annotate(
            int(y_max), (x_center, y_max),
            xytext=(0, 4), textcoords='offset points',
            ha='center', va='bottom'
        )
        y1_bbox = annot.get_window_extent().transformed(ax.transData.inverted()).y1
        y1_max_bboxes = max(y1_max_bboxes, y1_bbox)
    
    ax.set_ylim(top=y1_max_bboxes * 1.05)


def custom_chi2(
    df=None,
    var1=None,
    var2=None,
    crosstab=None,
    var_names=None,
    alpha=0.05,
    plot_contributions=False
):
    if any(arg is not None for arg in (df, var1, var2)) and crosstab is not None:
        raise ValueError(
            "Veuillez spécifier soit un dataframe et deux variables "
            "qualitatives, soit un tableau de contingence."
        )
    
    if crosstab is None:
        crosstab = pd.crosstab(index=df[var1], columns=df[var2])
    
    chi2, p, _, expected = stats.chi2_contingency(crosstab)
    
    sample_size = crosstab.sum().sum()
    min_dim = min(crosstab.shape)
    dof = min_dim - 1
    cramers_V = np.sqrt((chi2 / sample_size) / dof)
    cohens_omega = cramers_V * np.sqrt(dof)
    
    if cohens_omega < 0.1:
        effect_size = 'négligeable'
    elif cohens_omega < 0.3:
        effect_size = 'petite'
    elif cohens_omega < 0.5:
        effect_size = 'moyenne'
    else:
        effect_size = 'grande'

    expected = pd.DataFrame(
        data=expected,
        index=crosstab.index,
        columns=crosstab.columns,
    ).astype(int)
    
    text = f"La valeur du khi-2 est de {chi2:.2f} et la p-value de {p:.2g}. "
    if p < alpha:
        text += (
            f"Il existerait donc une liaison entre les variables \"{var1}\" et "
            f"\"{var2}\". La valeur du V de Cramer est de {cramers_V:.2g}, "
            f"indiquant que la taille de l'effet est {effect_size}."
        )
    else:
        text += (
            "On ne peut donc pas rejeter l'hypothèse H0 comme quoi les "
            f"variables sont indépendantes (au seuil alpha de {alpha})."
        )
    
    print("Tableau de contingence")
    display(crosstab)
    print()
    print("Effectifs attendus")
    display(expected)
    print()
    print(text)
    
    # contributions au khi-2
    if plot_contributions == True:
        fig, ax = plt.subplots(figsize=(6, 6))
        contributions_to_chi2 = ((crosstab - expected)**2 / expected) / chi2
        signs = (crosstab > expected) \
            .applymap(lambda x: '>' if x is True else '<')
        annot = np.round(contributions_to_chi2 * 100, 1).astype(str) + "%\n" + signs
        
        print("\n")
        print("Contributions au Khi-2")
        sns.heatmap(
            contributions_to_chi2,
            cmap='Blues',
            # annot=True,
            # fmt='.1%',
            annot=annot,
            fmt='',
            linewidth=1,
            ax=ax
        )
        # on modifie le nom des axes de la carte de chaleur
        if var_names is not None:
            xlabel, ylabel = ax.get_xlabel(), ax.get_ylabel()
            new_xlabel = var_names.get(xlabel, xlabel)
            new_ylabel = var_names.get(ylabel, ylabel)
            ax.set_xlabel(new_xlabel)
            ax.set_ylabel(new_ylabel)


def plot_vars(data, x_quanti='age', x_quali='age10', y_quanti=None, var_names=None, figsize=(16, 8)):
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flat
    
    sns.regplot(
        data=data,
        x=x_quanti,
        y=y_quanti,
        lowess=True,
        scatter_kws=dict(s=9, alpha=0.4),
        line_kws=dict(color='red'),
        ax=axes[0]
    )

    sns.boxplot(
        data=data,
        x=x_quanti,
        y=y_quanti,
        ax=axes[1]
    )

    for group in data[x_quali].unique().sort_values():
        group_data = data.loc[data[x_quali] == group]
        sns.regplot(
            data=group_data,
            x=x_quanti,
            y=y_quanti,
            lowess=True,
            scatter_kws=dict(s=9, alpha=0.4),
            line_kws=dict(color='red'),
            ax=axes[2]
        )

    sns.boxplot(
        data=data,
        x=x_quali,
        y=y_quanti,
        ax=axes[3]
    )

    # mise en forme des axes
    for ax in axes[[0, 2]]:
        ax.set_xlim(data[x_quanti].min(), data[x_quanti].max())
    axes[1].xaxis.set_major_locator(mticker.IndexLocator(base=5, offset=2.5))
    axes[1].grid(which='major', linestyle='-')
    
    # noms des axes des x et des y
    if var_names is not None:
        for ax in axes:
            xlabel, ylabel = ax.get_xlabel(), ax.get_ylabel()
            new_xlabel = var_names.get(xlabel, xlabel)
            new_ylabel = var_names.get(ylabel, ylabel)
            ax.set_xlabel(new_xlabel)
            ax.set_ylabel(new_ylabel)


def get_residuals(data, response, predictor):
    ols = smf.ols(response + '~' + predictor, data=data).fit()
    return ols.resid


@plt.rc_context({'axes.titleweight': 'bold'})
def check_normality_of_residuals(
    data,
    response,
    predictor,
    ks=False,
    hist_kws=None,
    alpha=0.05,
    axes=None
):  
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    
    if hist_kws is None:
        hist_kws = {}
        
    residuals = get_residuals(data, response, predictor)
    
    sns.histplot(
        residuals,
        stat='density',
        kde=True,
        label='Résidus',
        ax=axes[0],
        **hist_kws
    )

    x = np.linspace(residuals.min(), residuals.max(), 50)
    y = stats.norm.pdf(x, residuals.mean(), residuals.std())
    sns.lineplot(x=x, y=y, color='red', label='Distrib. normale', ax=axes[0])

    sm.qqplot(residuals, line='s', markersize=4, ax=axes[1])
    
    axes[0].set_title('Distribution des résidus')
    axes[1].set_title('Q-Q plot')
    
    plt.gcf().tight_layout()
    plt.show()
    
    if ks:
        ks_stat, ks_pvalue = stats.kstest(residuals, 'norm')
        print("\nKolmogorov-Smirnov")
        print("------------------")
        text = f"On obtient une statistique de {ks_stat:.2g} et une p-value de {ks_pvalue:.2g}. "
        if ks_pvalue < alpha:
            text += (
                f"La p-value étant inférieure au seuil alpha de {alpha}, on "
                "rejette l'hypothèse H0 comme quoi les résidus sont "
                "normalement distribués."
            )
        if ks_pvalue >= alpha:
            text += (
                f"La p-value étant supérieure au seuil alpha de {alpha}, on ne"
                "peut pas rejeter l'hypothèse H0 comme quoi les résidus sont "
                "normalement distribués."
            )
        print(text)


@plt.rc_context({'axes.titleweight': 'bold'})
def check_homoscedasticity_of_residuals(data, response, predictor, axes=None):
    if axes is not None:
        ax1, ax2 = axes
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    
    # on récupère les résidus et les valeurs prédites
    ols = smf.ols(response + '~' + predictor, data=data).fit()
    residuals = ols.resid
    fitted = ols.fittedvalues
    
    scatter_kws = dict(
        s=16,
        edgecolors='none',
        alpha=0.5
    )
    line_kws = dict(
        color='red'
    )
    
    # on représente les résidus vs. les valeurs prédites
    sns.regplot(
        x=fitted,
        y=residuals,
        lowess=True,
        scatter_kws=scatter_kws,
        line_kws=line_kws,
        ax=ax1
    )
    ax1.axhline(y=0, color='grey', linestyle='--', linewidth=1)
    ax1.set_xlabel('Fitted value')
    ax1.set_ylabel('Residual')
    ax1.set_title('Résidus vs. valeurs prédites')
    
    # on représente la valeur absolue des résidus vs. les valeurs prédites
    sns.regplot(
        x=fitted,
        y=np.abs(residuals),
        lowess=True,
        scatter_kws=scatter_kws,
        line_kws=line_kws,
        ax=ax2
    )
    # ax.axhline(y=0, color='grey', linestyle='--', linewidth=1)
    ax2.set_xlabel('Fitted value')
    ax2.set_ylabel('Absolute value of residual')
    ax2.set_title('Valeurs absolues des résidus vs. valeurs prédites')
    
    plt.gcf().tight_layout()
    plt.show()


def check_normality_and_homoscedasticity_of_residuals(data, response, predictor, ks=False):
    print("Normalité des résidus")
    print("=====================")
    check_normality_of_residuals(data, response, predictor, ks=ks)
    print("\n") if ks else print()
    print("Homoscédasticité des résidus")
    print("============================")
    check_homoscedasticity_of_residuals(data, response, predictor)


def custom_anova(data, response, predictor, alpha=0.05):
    ols = smf.ols(response + '~' + predictor, data=data).fit()
    f_pvalue = ols.f_pvalue
    print("ANOVA")
    print("-----")
    print(
        f"On obtient une statistique F de {ols.fvalue:.1f} et une p-value de "
        f"{f_pvalue:.3g}."
    )
    if f_pvalue < alpha:
        print(
            f"La p-value étant inférieure au seuil alpha de {alpha}, on "
            "rejette l'hypothèse H0 : les moyennes sont globalement "
            "différentes."
        )
        print(
            f"Le R carré ajusté est de {ols.rsquared_adj:.3f}, ce qui signifie "
            f"que {ols.rsquared_adj * 100:.1f} % de la variation est expliquée "
            f"par la variable \"{predictor}\"."
        )
    print()
    return sm.stats.anova_lm(ols, typ=2)


def custom_kruskal(data, response, predictor, alpha=0.05):
    groups = data[predictor].unique()
    group_data = [data.loc[data[predictor] == group, response] for group in groups]
    
    stat, pvalue = stats.kruskal(*group_data)
    
    epsilon_squared = stat / (data[response].size - 1)
    if epsilon_squared < 0.01:
        effect_size = 'négligeable'
    elif epsilon_squared < 0.08:
        effect_size = 'petite'
    elif epsilon_squared < 0.26:
        effect_size = 'moyenne'
    else:
        effect_size = 'grande'
    
    print("Kruskal-Wallis")
    print("--------------")
    print(
        f"On obtient une statistique H de {stat:.1f} et une p-value de "
        f"{pvalue:.3g}."
    )
    if pvalue < alpha:
        print(
            f"La p-value étant inférieure au seuil alpha de {alpha}, on "
            "rejette l'hypothèse H0 comme quoi les médianes sont égales. "
            "Au moins une des médianes est différente des autres."
        )
        print(
            f"La valeur de l'epsilon carré (équivalent de l'êta carré dans une "
            f"ANOVA) est de {epsilon_squared:.3g}, indiquant que la taille de "
            f"l'effet est {effect_size}.")
    else:
        print(
            f"La p-value étant inférieure au seuil alpha de {alpha}, on "
            "ne peut pas rejeter l'hypothèse H0 comme quoi les médianes sont "
            "égales."
        )


def custom_dunn(data, response, predictor='age10', palette='Blues_r', figsize=(6, 6)):   
    data = data.sort_values(by=predictor)
    groups = data[predictor].unique()
    
    pvalues = sp.posthoc_dunn(
        a=data,
        val_col=response,
        group_col=predictor,
        p_adjust='bonferroni'
    )
    
    pvalues.index = pvalues.columns = groups
    
    fig, ax = plt.subplots(figsize=figsize)
    cmap = mcolors.ListedColormap(sns.color_palette(palette, n_colors=3) + ['lightgrey'])
    boundaries = [0, 0.001, 0.01, 0.05, 1]
    norm = mcolors.BoundaryNorm(boundaries=boundaries, ncolors=cmap.N)

    sns.heatmap(pvalues,
                annot=True,
                fmt='.3f',
                cmap=cmap,
                norm=norm,
                square=True,
                linewidths=1)

    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([np.mean(x) for x in zip(boundaries, boundaries[1:])])
    colorbar.set_ticklabels(['p < 0.001', '0.001 <= p < 0.01', '0.01 <= p < 0.05', 'p >= 0.05'])
    
    ax.set_title("p-values", weight='bold')
    
    print("Test post-hoc de Dunn")
    print("---------------------")
    plt.show()
    
    
def custom_tukey_hsd(data, response, predictor='age10', palette='Blues_r', figsize=(6, 6)):
    data = data.sort_values(by=predictor)
    groups = data[predictor].unique()
    
    group_data = [data.loc[data[predictor] == group, response] for group in groups]
    
    res = stats.tukey_hsd(*group_data)
    pvalues = pd.DataFrame(data=res.pvalue, index=groups, columns=groups)
    
    fig, ax = plt.subplots(figsize=figsize)
    cmap = mcolors.ListedColormap(sns.color_palette(palette, n_colors=3) + ['lightgrey'])
    boundaries = [0, 0.001, 0.01, 0.05, 1]
    norm = mcolors.BoundaryNorm(boundaries=boundaries, ncolors=cmap.N)

    sns.heatmap(pvalues,
                annot=True,
                fmt='.3f',
                cmap=cmap,
                norm=norm,
                square=True,
                linewidths=1)

    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([np.mean(x) for x in zip(boundaries, boundaries[1:])])
    colorbar.set_ticklabels(['p < 0.001', '0.001 <= p < 0.01', '0.01 <= p < 0.05', 'p >= 0.05'])
    
    ax.set_title("p-values", weight='bold')
    
    print("Test post-hoc de Tukey HSD")
    print("--------------------------")
    plt.show()


@plt.rc_context({'axes.labelweight': 'bold'})
def boxplot_letters(
    data,
    response,
    predictor,
    letters,
    var_names=None,
    ax=None
):
    if ax is None:
        ax = plt.gca()
        
    meanprops = {
        'marker':'o',
        'markerfacecolor':'white', 
        'markeredgecolor':'black'
    }
    
    sns.boxplot(x=data[predictor],
                y=data[response],
                showmeans=True,
                meanprops=meanprops,
                ax=ax)

    n_patches = len(ax.patches)
    lines = np.array(ax.lines).reshape(n_patches, -1)

    # on cache les outliers
    for outlier in lines[:, 6]:
        outlier.set(alpha=0)

    plt.draw()
    y1_max = 0

    # on annote les moustaches supérieures avec les lettres
    for i, whisker in enumerate(lines[:, 1]):
        x, y = whisker.get_xydata()[1]
        annot = ax.annotate(
            text=letters[i],
            xy=(x, y),
            xytext=(0, 6),
            textcoords='offset points',
            ha='center'
        )
        y1_bbox = annot.get_window_extent().transformed(ax.transData.inverted()).y1
        y1_max = max(y1_max, y1_bbox)

    ax.set_ylim(top=y1_max * 1.05)
    
    # on affiche les effectifs des groupes
    counts = data.groupby(predictor, as_index=False).agg(count=(response, 'count')).to_numpy()
    group_counts = {group: str(count) for group, count in counts}
    labels = [
        label.get_text() + '\n(' + group_counts[label.get_text()] + ')'
        for label in ax.get_xticklabels()
    ]
    ax.set_xticklabels(labels)
    
    # on modifie le nom des axes des x et des y
    if var_names is not None:
        xlabel, ylabel = ax.get_xlabel(), ax.get_ylabel()
        new_xlabel = var_names.get(xlabel, xlabel)
        new_ylabel = var_names.get(ylabel, ylabel)
        ax.set_xlabel(new_xlabel)
        ax.set_ylabel(new_ylabel)
    
    plt.show()