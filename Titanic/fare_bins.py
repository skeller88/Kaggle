import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab as P
import scipy
import statsmodels.api as sm

def fare_bins_heuristic():
    """
    Extends on the Kaggle survival table solution by binning by fare quantiles.
    """

    def add_bin_fare(frame, number_of_bins=10):
        """Bin the ticket fare and add new column FareBin"""
        p_classes = sorted(frame.Pclass.unique())

        fare_avgs = [frame[['Fare']][frame['Pclass'] == p_class].mean()
                     for p_class in p_classes]

        fare_avgs_by_class = pd.Series(fare_avgs, index=p_classes)

        for p_class in p_classes:
            frame.loc[
                (np.isnan(frame['Fare'])) &
                (frame['Pclass'] == p_class)] = fare_avgs_by_class[p_class - 1]

            print '-------------', fare_avgs_by_class[p_class - 1]

        frame['FareBin'] = ((frame.Fare // fare_bracket_size)
            .clip_upper(number_of_fares - 1)
            # .astype(np.int)
        )

    train = pd.read_csv('Titanic/train.csv')

    fare_ceiling = 40
    fare_bracket_size = 10
    number_of_fares = fare_ceiling // fare_bracket_size
    number_of_classes = len(np.unique(train.Pclass))

    add_bin_fare(train, fare_bracket_size=fare_bracket_size,
                 number_of_fares=number_of_fares)

    index_list = []
    survival_count = []
    survival_list = []
    gender_names = ['female', 'male']

    def get_obs_for_categories(sex_idx, class_idx, fare_idx):
        return train.Survived[(train.Sex == gender_names[sex_idx])
                              & (train.Pclass - 1 == class_idx)
                              & (train.FareBin == fare_idx)]

    for sex_idx in xrange(2):
        for class_idx in xrange(number_of_classes):
            for fare_idx in xrange(number_of_fares):
                index_list += [(sex_idx, class_idx, fare_idx)]
                survival_probability = (get_obs_for_categories(
                    sex_idx,
                    class_idx,
                    fare_idx)
                    .mean())
                survival_list.append(survival_probability)
                survival_count.append(len(get_obs_for_categories(
                    sex_idx,
                    class_idx,
                    fare_idx))
                )

    survival_index = pd.MultiIndex.from_tuples(index_list, names=[
        'Gender',
        'Class',
        'FareBin'
    ])

    survival_table = (pd.Series(survival_list,
                               index=survival_index,
                               name='Survival')
                      .fillna(0))

    print survival_table, survival_count

    test = pd.read_csv('Titanic/test.csv', index_col=[0])

    add_bin_fare(test)

    def is_survivor(series):
        """series format: ['Sex', 'Pclass', 'FareBin']"""

        sex_idx = 1 if series.Sex == 'male' else 0
        index = (sex_idx, series.Pclass - 1, series.FareBin)
        return survival_table[index]

    test['Survived'] = test[['Sex', 'Pclass', 'FareBin']].apply(is_survivor,
                                                             axis=1)

    test[['Survived']].to_csv('genderclassmodel_with_binomial_probs.csv')

fare_bins_heuristic()