# Standard libraries
from itertools import product

# Data manipulation and analysis
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from pandas import DataFrame, cut

# Visualization
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns

# Custom config, constants and utils
from config import (
    img_full_pitch, img_half_pitch, PASS_LINE_COLORS,
    PITCH_LENGTH, PITCH_WIDTH
)
from utils import filter_df, n_bins_labels


def category_heatmap(
    df: DataFrame,
    feature_1: str,
    feature_2: str,
    bin_count_1: int = 5,
    bin_count_2: int = 5,
    equal_bin_sizes_1: bool = False,
    equal_bin_sizes_2: bool = False
) -> None:
    feature_names = {}
    categories = {}

    for feature, bin_count, equal_bin_sizes in zip(
        [feature_1, feature_2],
        [bin_count_1, bin_count_2],
        [equal_bin_sizes_1, equal_bin_sizes_2]
    ):
        if is_numeric_dtype(df[feature]):
            bins, labels = n_bins_labels(
                df[feature],
                bin_count,
                equal_bin_sizes
            )
            df[f'{feature}Bins'] = cut(
                df[feature],
                bins=bins,
                labels=labels,
                right=True,
                include_lowest=True
            )
            feature_names[feature] = f'{feature}Bins'
            categories[feature] = labels
        else:
            feature_names[feature] = feature
            categories[feature] = df[f'{feature}'].unique()

    share_matrix = [
        [
            df[
                (df[feature_names[feature_1]] == val_1) &
                (df[feature_names[feature_2]] == val_2)
            ].shape[0] / df.shape[0] * 100
            for val_1 in categories[feature_1]
        ] for val_2 in categories[feature_2]
    ]

    share_matrix = DataFrame(share_matrix).dropna()
    annot = [
        [
            f'{share_matrix.iloc[i, j]:.2f}%'
            for j in range(share_matrix.shape[1])
        ] for i in range(share_matrix.shape[0])
    ]

    ax = sns.heatmap(
        data=share_matrix,
        annot=annot,
        cbar=False,
        cmap='coolwarm',
        fmt='',
        square=True
    )

    ax.set_xticklabels(categories[feature_1], rotation=45, ha='right')
    ax.set_yticklabels(categories[feature_2], rotation=0)
    plt.xlabel(feature_1)
    plt.ylabel(feature_2)
    plt.title(f'Share of shots by {feature_1} and {feature_2}')

    return None


def feature_goal_corr_bars(
    df: DataFrame,
    feature: str,
    bin_count: int = 5,
    equal_bin_sizes: bool = True
) -> None:
    if is_numeric_dtype(df[feature]):
        bins, labels = n_bins_labels(df[feature], bin_count, equal_bin_sizes)
        df[f'{feature}Bins'] = cut(
            df[feature],
            bins=bins,
            labels=labels,
            right=True,
            include_lowest=True
        )
        feature_counts = df[f'{feature}Bins'].value_counts()
        goal_shares = df.groupby(f'{feature}Bins')['goal'].mean()
    else:
        feature_counts = df[feature].value_counts()
        goal_shares = df.groupby(feature)['goal'].mean()
        if not is_bool_dtype(df[feature]):
            goal_shares.sort_values(ascending=False, inplace=True)
            feature_counts = feature_counts.loc[goal_shares.index]

    ax = sns.barplot(x=goal_shares.index, y=goal_shares)
    sns.despine(right=True, top=True)

    for bar, count in zip(ax.patches, feature_counts):
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height() / 2
        annotation = f'{y * 2 * 100:.2f}%\n(out of {count})'
        plt.annotate(annotation, (x, y), ha='center', va='center')

    plt.xticks(rotation=45, ha='right')
    plt.ylabel('goal rate')
    plt.title(f'Goal rate by {feature}')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    return None


def plot_passes(
    df_events: DataFrame,
    df_players: DataFrame,
    df_matches: DataFrame,
    player_id: int,
    match_id: int,
    feature: str = 'angleCategory',
) -> None:

    df_player_passes = filter_df(
        df_events,
        matchId=match_id,
        playerId=player_id,
        eventName='Pass'
    )

    unique_values = df_player_passes[feature].unique()
    colors = plt.colormaps['tab10'].resampled(len(unique_values)).colors
    color_map = dict(
        zip(unique_values, [mcolors.rgb2hex(c) for c in colors])
    )

    pass_lines = [
        [
            [
                row['xPosStart'] * PITCH_LENGTH / 100,
                (100 - row['yPosStart']) * PITCH_WIDTH / 100
            ],
            [
                row['xPosEnd'] * PITCH_LENGTH / 100,
                (100 - row['yPosEnd']) * PITCH_WIDTH / 100
            ]
        ]
        for _, row in df_player_passes.iterrows()
    ]

    if feature == 'angleCategory':
        colors = [
            PASS_LINE_COLORS[row['angleCategory']]
            for _, row in df_player_passes.iterrows()
        ]
    else:
        colors = [
            color_map[row[feature]] for _, row in df_player_passes.iterrows()
        ]

    linestyles = [
        '-' if row['accurate'] else ':'
        for _, row in df_player_passes.iterrows()
    ]

    lc = LineCollection(
        pass_lines,
        colors=colors,
        linestyles=linestyles,
        linewidths=1.5
    )
    plt.gca().add_collection(lc)

    start_x = df_player_passes['xPosStart'] * PITCH_LENGTH / 100
    start_y = (100 - df_player_passes['yPosStart']) * PITCH_WIDTH / 100
    end_x = df_player_passes['xPosEnd'] * PITCH_LENGTH / 100
    end_y = (100 - df_player_passes['yPosEnd']) * PITCH_WIDTH / 100

    plt.scatter(
        start_x,
        start_y,
        marker='o',
        s=30,
        c=colors,
        zorder=5
    )

    plt.scatter(
        end_x,
        end_y,
        marker='x',
        s=30,
        c=colors,
        zorder=5
    )

    plt.imshow(
        img_full_pitch,
        extent=[0, 100, 0, 65],
        aspect='auto'
    )
    plt.xlim(0, 100)
    plt.ylim(0, 65)
    plt.axis('off')

    match_label = df_matches.loc[match_id, 'label']
    player_name = df_players.loc[player_id, 'shortName']
    plt.title(
        f'Passes of {player_name} in match {match_label}'
    )

    linestyle_legend = [
        Line2D(
            [0],
            [0],
            color='black',
            lw=2,
            linestyle='-',
            label='Accurate'
        ),
        Line2D(
            [0],
            [0],
            color='black',
            lw=2,
            linestyle=':',
            label='Not accurate'
        )
    ]

    accuracy_legend = plt.legend(
        handles=linestyle_legend,
        title='Accuracy',
        bbox_to_anchor=(1, 1),
        loc='upper left'
    )
    plt.gca().add_artist(accuracy_legend)

    scatter_legend = [
        Line2D(
            [0],
            [0],
            marker='o',
            color='black',
            markerfacecolor='black',
            markersize=5,
            label='Startpoint',
            linestyle='None'
        ),
        Line2D(
            [0],
            [0],
            marker='x',
            color='black',
            markerfacecolor='black',
            markersize=5,
            label='Endpoint',
            linestyle='None'
        )
    ]
    position_legend = plt.legend(
        handles=scatter_legend,
        title='Pass Points',
        bbox_to_anchor=(1, 0.8),
        loc='upper left'
    )
    plt.gca().add_artist(position_legend)

    if feature == 'angleCategory':
        color_legend = [
            Line2D([0], [0], color=PASS_LINE_COLORS[k], lw=2, label=k)
            for k in PASS_LINE_COLORS
        ]
    else:
        color_legend = [
            Line2D([0], [0], color=color_map[k], lw=2, label=str(k))
            for k in color_map
        ]

    plt.legend(
        handles=color_legend,
        title=feature,
        bbox_to_anchor=(1, 0.6),
        loc='upper left'
    )
    return None


def plot_passing_stats(
    df_events: DataFrame,
    df_players: DataFrame,
    df_matches: DataFrame,
    player_id: int,
    match_id: int = None
) -> None:
    df_player_passes = filter_df(
        df_events,
        matchId=match_id,
        playerId=player_id,
        eventName='Pass'
    )

    accurate_passes = df_player_passes[df_player_passes['accurate']]
    direction_shares_accurate = (
        accurate_passes.groupby(
            'angleCategory', observed=False
        )['id'].count() /
        df_player_passes.shape[0]
    )

    inaccurate_passes = df_player_passes[
        df_player_passes['accurate'].eq(False)
    ]
    direction_shares_inaccurate = (
        inaccurate_passes.groupby(
            'angleCategory', observed=False
        )['id'].count() /
        df_player_passes.shape[0]
    )

    angle_category_order = [
        'vertical backward',
        'diagonal backward',
        'horizontal',
        'diagonal forward',
        'vertical forward'
    ]

    direction_shares_accurate = direction_shares_accurate.reindex(
        angle_category_order,
        fill_value=0
    )
    direction_shares_inaccurate = direction_shares_inaccurate.reindex(
        angle_category_order,
        fill_value=0
    )

    sns.barplot(
        x=direction_shares_accurate.index,
        y=direction_shares_accurate,
        color='mediumseagreen'
    )
    sns.barplot(
        x=direction_shares_accurate.index,
        y=direction_shares_inaccurate,
        bottom=direction_shares_accurate,
        color='lightcoral'
    )
    sns.despine(right=True, top=True)

    player_name = df_players.loc[player_id, 'shortName']
    match_label = df_matches.loc[match_id, 'label'] if match_id else ''
    plt.title(
        f'Passing-directions of {player_name}' +
        (
            f' in match {match_label}' if match_id else ''
        ) +
        f' ({df_player_passes.shape[0]} passes)'
    )
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('share of passes')
    plt.legend(
        handles=[
            plt.Rectangle(
                (0, 0),
                1,
                1,
                color='mediumseagreen',
                label='Accurate'
            ),
            plt.Rectangle(
                (0, 0),
                1,
                1,
                color='lightcoral',
                label='Inaccurate'
            )
        ],
        title='Pass Accuracy'
    )
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    for i, values in enumerate(
        zip(direction_shares_accurate, direction_shares_inaccurate)
    ):
        for j in range(2):
            if values[j] != 0:
                plt.annotate(
                    f'{values[j] * 100:.1f}%',
                    (i, values[j] / 2 + values[0] * j),
                    ha='center',
                    va='center'
                )
        plt.annotate(
            f'∑ {(values[0] + values[1]) * 100:.1f}%',
            (i, values[0] + values[1] + 0.01),
            ha='center',
            va='center',
            fontweight='bold'
        )
    return None


def plot_shots(
    df_shots: DataFrame,
    df_players: DataFrame,
    df_teams: DataFrame,
    df_matches: DataFrame,
    feature: str,
    show_misses: bool = True,
    match_id: int = None,
    team_id: int = None,
    player_id: int = None
) -> None:

    df_relevant_shots = filter_df(
        df_shots,
        matchId=match_id,
        teamId=team_id,
        playerId=player_id
    )

    if show_misses:
        df_misses = df_relevant_shots[~df_relevant_shots['goal']].copy()
    df_goals = df_relevant_shots[df_relevant_shots['goal']].copy()

    plt.imshow(
        img_half_pitch,
        extent=[50, 100, 0, 100],
        aspect=0.5 * PITCH_LENGTH / PITCH_WIDTH
    )
    plt.axis('off')

    unit_name = (
        df_players.loc[player_id, 'shortName'] if player_id
        else (df_teams.loc[team_id, 'name'] if team_id else None)
    )
    unit_name = f'of {unit_name}' if unit_name else ''
    match_label = df_matches.loc[match_id, 'label']

    plt.title(
        f'{'Shots' if show_misses else 'Goals'} '
        f'{unit_name} in match\n{match_label}'
    )

    if is_numeric_dtype(df_relevant_shots[feature]):
        vmin = df_relevant_shots[feature].min()
        vmax = df_relevant_shots[feature].max()
        if show_misses:
            plt.scatter(
                x='xPosStart',
                y='yPosStart',
                data=df_misses,
                c=df_misses[feature],
                cmap='inferno',
                vmin=vmin,
                vmax=vmax,
                marker='o'
            )
            plt.colorbar(label=feature)
        plt.scatter(
            x='xPosStart',
            y='yPosStart',
            data=df_goals,
            c=df_goals[feature],
            cmap='inferno',
            vmin=vmin,
            vmax=vmax,
            marker='x'
        )
        if not show_misses:
            plt.colorbar(label=feature)
    else:
        categories = df_relevant_shots[feature].unique()
        color_map = {
            category: plt.cm.tab10(i)
            for i, category in enumerate(categories)
        }
        if show_misses:
            for category, color in color_map.items():
                subset = df_misses[df_misses[feature] == category]
                plt.scatter(
                    x='xPosStart',
                    y='yPosStart',
                    data=subset,
                    label=category,
                    color=color,
                    marker='o'
                )
            plt.legend(title=feature, loc='upper left')
        for category, color in color_map.items():
            subset = df_goals[df_goals[feature] == category]
            plt.scatter(
                x='xPosStart',
                y='yPosStart',
                data=subset,
                label=category,
                color=color,
                marker='x'
            )
        if not show_misses:
            plt.legend(title=feature, loc='upper left')
    return None


def two_features_goal_corr_bars(
    df: DataFrame,
    feature_1: str,
    feature_2: str,
    bin_count_1: int = 5,
    bin_count_2: int = 5,
    equal_bin_sizes_1: bool = False,
    equal_bin_sizes_2: bool = False
) -> None:
    feature_names = {}
    categories = {}

    for feature, bin_count, equal_bin_sizes in zip(
        [feature_1, feature_2],
        [bin_count_1, bin_count_2],
        [equal_bin_sizes_1, equal_bin_sizes_2]
    ):
        if is_numeric_dtype(df[feature]):
            bins, labels = n_bins_labels(
                df[feature],
                bin_count,
                equal_bin_sizes
            )
            df[f'{feature}Bins'] = cut(
                df[feature],
                bins=bins,
                labels=labels,
                right=True,
                include_lowest=True
            )
            feature_names[feature] = f'{feature}Bins'
            categories[feature] = labels
        else:
            feature_names[feature] = feature
            categories[feature] = df[f'{feature}'].unique()

    goal_share_matrix = [
        [
            val_1,
            val_2,
            df[
                (df[feature_names[feature_1]] == val_1) &
                (df[feature_names[feature_2]] == val_2)
            ]['goal'].mean()
        ] for val_1, val_2 in product(
            categories[feature_1],
            categories[feature_2]
        )
    ]

    df_goal_shares = DataFrame(
        goal_share_matrix,
        columns=[feature_1, feature_2, 'goalRate']
    ).dropna()

    ax = sns.barplot(
        data=df_goal_shares,
        x=feature_1,
        y='goalRate',
        hue=feature_2
    )
    sns.despine(right=True, top=True)
    ax.set_xticks(range(len(categories[feature_1])))
    ax.set_xticklabels(
        labels=categories[feature_1],
        rotation=45,
        ha='right'
    )
    plt.title(
        f'Goal rate by {feature_1} and {feature_2}'
    )
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    return None
