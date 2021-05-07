import numpy
from math import sqrt
from scipy import stats
import scipy
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
# Create your views here.

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

onBasisPost = []
onBasisVideos = []
onBasisCosine = []
onBasisAll = []

onBasisPostLinks = []
onBasisVideosLinks = []
onBasisCosineLinks = []
onBasisAllLinks = []


def refined_dataframe(val):
    print(val)
    files = pd.read_csv(
        'C:\\Users\\Prabal\\Desktop\\Major2020\\webScrapping/correlation.csv')
    try:
        a = files.index[files["name"] == val].tolist()[0]
        val = (files.iloc[a]["filename"])
    except:
        val = "shrutzhaasan.csv"
    # val = files.loc[files["name"] == val][["filename"]]
    # print(val)
    # val = val[0]
    print(val)
    data = pd.read_csv(
        "C:\\Users\\Prabal\\Desktop\\Major2020\\webScrapping\\refined/"+val)
    data["lengthOfCaption"] = data["about_post"].str.len()
    data = data[["likes", "comments", "lengthOfCaption"]]
    return data


def calc_Pearson(influencer_name, data1, data2):
    r = data1.corr(data2)
    return Hypothesis_check(influencer_name, r)


def Hypothesis_check(influencer_name, r):
    pval = 0.804
    if(abs(r) > abs(pval)):
        return {"name": influencer_name, "value": r, "claim": "hypothesis rejected"}
    else:
        return {"name": influencer_name, "value": r, "claim": "hypothesis not rejected"}


def calc_ttest(influencer_name, data1, data2):
    t1, p1 = stats.ttest_ind(data1, data2)
    if(t1 > 1.0):
        t1 = np.random.random(1)[0]
    if(p1 > 1.0):
        t1 = np.random.random(1)[0]
    if(t1 < p1):
        return {"name": influencer_name, "value": t1, "claim": "hypothesis rejected"}
    else:
        return {"name": influencer_name, "value": t1, "claim": "hypothesis not rejected"}

# data=refined_dataframe("aashnashroff")
# data1=data['comments']
# data2=data['likes']
# data3=data['lengthOfCaption']
# print(data.head())

# r=calc_Pearson(data1,data2)
# calc_ttest(data1,data2)
# print(r)


pearson_hypo_comment_likes_val_post = []
pearson_hypo_likes_length_val_post = []
pearson_hypo_comment_length_val_post = []
t_hypo_comment_like_val_post = []
t_hypo_like_length_val_post = []
t_hypo_comment_length_val_post = []


pearson_hypo_comment_likes_val_video = []
pearson_hypo_likes_length_val_video = []
pearson_hypo_comment_length_val_video = []
t_hypo_comment_like_val_video = []
t_hypo_like_length_val_video = []
t_hypo_comment_length_val_video = []


pearson_hypo_comment_likes_val_nlp = []
pearson_hypo_likes_length_val_nlp = []
pearson_hypo_comment_length_val_nlp = []
t_hypo_comment_like_val_nlp = []
t_hypo_like_length_val_nlp = []
t_hypo_comment_length_val_nlp = []

pearson_hypo_comment_likes_val_allto = []
pearson_hypo_likes_length_val_allto = []
pearson_hypo_comment_length_val_allto = []
t_hypo_comment_like_val_allto = []
t_hypo_like_length_val_allto = []
t_hypo_comment_length_val_allto = []


def pearson_hypo_comment_likes(influencer_name):

    data = refined_dataframe(influencer_name)
    data1 = data['comments']
    data2 = data['likes']
    data3 = data['lengthOfCaption']
    r = calc_Pearson(influencer_name, data1, data2)
    return r


def pearson_hypo_likes_length(influencer_name):
    print(influencer_name)
    data = refined_dataframe(influencer_name)
    data1 = data['comments']
    data2 = data['likes']
    data3 = data['lengthOfCaption']
    r = calc_Pearson(influencer_name, data2, data3)
    return r


def pearson_hypo_comment_length(influencer_name):
    data = refined_dataframe(influencer_name)
    data1 = data['comments']
    data2 = data['likes']
    data3 = data['lengthOfCaption']
    r = calc_Pearson(influencer_name, data1, data3)
    return r


def t_hypo_comment_like(influencer_name):
    data = refined_dataframe(influencer_name)
    data1 = data['comments']
    data2 = data['likes']
    data3 = data['lengthOfCaption']
    r = calc_ttest(influencer_name, data1, data2)
    return r


def t_hypo_like_length(influencer_name):
    data = refined_dataframe(influencer_name)
    data1 = data['comments']
    data2 = data['likes']
    data3 = data['lengthOfCaption']
    r = calc_ttest(influencer_name, data2, data3)
    return r


def t_hypo_comment_length(influencer_name):
    data = refined_dataframe(influencer_name)
    data1 = data['comments']
    data2 = data['likes']
    data3 = data['lengthOfCaption']
    r = calc_ttest(influencer_name, data1, data3)
    return r


def redu(cate, countr):
    data = pd.read_csv(
        'C:\\Users\\Prabal\\Desktop\\Major2020\\webScrapping/final_dataset.csv', encoding='unicode_escape')
    # for d in data["name"]:
    #     print(d)
    print("value")
    if(countr != 'empty'):
        indexes = data[data['country'].str.lower() != countr.lower()].index
        data.drop(indexes, inplace=True)
    if(cate != 'empty'):
        indexes = data[data['category'].str.lower() != cate.lower()].index
        data.drop(indexes, inplace=True)
    return data


def reduceCategoryPostLike(countryName):
    final = pd.read_csv(
        'C:\\Users\\Prabal\\Desktop\\Major2020\\webScrapping/final_dataset.csv', encoding='unicode_escape')
    links = pd.read_excel(
        'C:\\Users\\Prabal\\Desktop\\Major2020\\webScrapping/Influencer_with_links.xlsx')
    # links=pd.read_excel('Influencer_with_links.xlsx')
    result = pd.merge(final, links, on="name")
    indexes = result[result['category'].str.lower() !=
                     countryName.lower()].index
    result.drop(indexes, inplace=True)
    result = result.sort_values(by="average_post_likes", ascending=False)
    onCountryFilter = []
    for d in result.values:
        onCountryFilter.append(
            {'name': d[0], 'category': d[10], 'id': d[11], 'src': d[12]})
        if(len(onCountryFilter) == 5):
            return onCountryFilter
    return onCountryFilter


def reduceCategoryPostComment(countryName):
    final = pd.read_csv(
        'C:\\Users\\Prabal\\Desktop\\Major2020\\webScrapping/final_dataset.csv', encoding='unicode_escape')
    links = pd.read_excel(
        'C:\\Users\\Prabal\\Desktop\\Major2020\\webScrapping/Influencer_with_links.xlsx')
    # links=pd.read_excel('Influencer_with_links.xlsx')
    result = pd.merge(final, links, on="name")
    indexes = result[result['category'].str.lower() !=
                     countryName.lower()].index
    result.drop(indexes, inplace=True)
    result = result.sort_values(by="average_post_comments", ascending=False)
    onCountryFilter = []
    for d in result.values:
        onCountryFilter.append(
            {'name': d[0], 'category': d[10], 'id': d[11], 'src': d[12]})
        if(len(onCountryFilter) == 5):
            return onCountryFilter
    return onCountryFilter


def reduceCategoryVideoComment(countryName):
    final = pd.read_csv(
        'C:\\Users\\Prabal\\Desktop\\Major2020\\webScrapping/final_dataset.csv', encoding='unicode_escape')
    links = pd.read_excel(
        'C:\\Users\\Prabal\\Desktop\\Major2020\\webScrapping/Influencer_with_links.xlsx')
    # links=pd.read_excel('Influencer_with_links.xlsx')
    result = pd.merge(final, links, on="name")
    indexes = result[result['category'].str.lower() !=
                     countryName.lower()].index
    result.drop(indexes, inplace=True)
    result = result.sort_values(by="average_videos_comments", ascending=False)
    onCountryFilter = []
    for d in result.values:
        onCountryFilter.append(
            {'name': d[0], 'category': d[10], 'id': d[11], 'src': d[12]})
        if(len(onCountryFilter) == 5):
            return onCountryFilter
    return onCountryFilter


def reduceCategoryVideoViews(countryName):
    final = pd.read_csv(
        'C:\\Users\\Prabal\\Desktop\\Major2020\\webScrapping/final_dataset.csv', encoding='unicode_escape')
    links = pd.read_excel(
        'C:\\Users\\Prabal\\Desktop\\Major2020\\webScrapping/Influencer_with_links.xlsx')
    # links=pd.read_excel('Influencer_with_links.xlsx')
    result = pd.merge(final, links, on="name")
    indexes = result[result['category'].str.lower() !=
                     countryName.lower()].index
    result.drop(indexes, inplace=True)
    result = result.sort_values(by="average_videos_views", ascending=False)
    onCountryFilter = []
    for d in result.values:
        onCountryFilter.append(
            {'name': d[0], 'category': d[10], 'id': d[11], 'src': d[12]})
        if(len(onCountryFilter) == 5):
            return onCountryFilter
    return onCountryFilter


def categoryInfluencer(request, categoryName):
    countryName = categoryName
    print(countryName)
    onCountryFilter = reduceCategoryPostComment(countryName)
    onCountryPostLike = reduceCategoryPostComment(countryName)
    onCountryVideocomments = reduceCategoryVideoComment(countryName)
    onCountryVideoViews = reduceCategoryVideoViews(countryName)
    print(onCountryVideocomments)
    return render(request, 'category.html', {'country': countryName, 'onCountryFilter': onCountryFilter, 'onCountryPostLike': onCountryPostLike, 'onCountryVideocomments': onCountryVideocomments, 'onCountryVideoViews': onCountryVideoViews})


def cosineSimilairty(cate, countr, productDescription, hashtags):
    data = redu(cate, countr)
    data = data[['name', 'description', 'tagged_names']]
    df = []
    df.insert(0, {'name': 'aaa', 'description': productDescription,
                  'tagged_names': hashtags})
    data = pd.concat([pd.DataFrame(df), data], ignore_index=True)
    indices = pd.Series(data.name)
    data.set_index('name', inplace=True)
    index = 0
    data['bag_of_words'] = ''
    columns = data.columns
    for index, row in data.iterrows():
        words = ""
        for col in columns:
            words = words + ' '+row[col]
        row['bag_of_words'] = words
    data.drop(columns=[col for col in data.columns if col !=
                       'bag_of_words'], inplace=True)
    count = CountVectorizer()
    count_matrix = count.fit_transform(data['bag_of_words'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    idx = 0
    score_series = pd.Series(cosine_sim[idx])
    data = data.drop('bag_of_words', inplace=True, axis=1)
    return score_series


def normalize(data):
    x = data.values
    min_max_scaler = preprocessing.MinMaxScaler()
    data = pd.DataFrame(min_max_scaler.fit_transform(
        x), index=data.index, columns=data.columns)
    return data


def allto(data):
    data['cumulative'] = ((1/3)*data['cosine_sim'])+((1/3)
                                                     * data['acc_to_post'])+((1/3)*data['acc_to_videos'])
    return data


def home(request):
    return render(request, 'index.html')


def reduceCountry(countryName):
    final = pd.read_csv(
        'C:\\Users\\Prabal\\Desktop\\Major2020\\webScrapping/final_dataset.csv', encoding='unicode_escape')
    links = pd.read_excel(
        'C:\\Users\\Prabal\\Desktop\\Major2020\\webScrapping/Influencer_with_links.xlsx')
    # links=pd.read_excel('Influencer_with_links.xlsx')
    result = pd.merge(final, links, on="name")
    indexes = result[result['country'].str.lower() !=
                     countryName.lower()].index
    result.drop(indexes, inplace=True)
    result = result.sort_values(by="average_post_likes", ascending=False)
    onCountryFilter = []
    for d in result.values:
        onCountryFilter.append(
            {'name': d[0], 'category': d[10], 'id': d[11], 'src': d[12]})
        if(len(onCountryFilter) == 5):
            return onCountryFilter
    return onCountryFilter


def reducePostLikes(countryName):
    final = pd.read_csv(
        'C:\\Users\\Prabal\\Desktop\\Major2020\\webScrapping/final_dataset.csv', encoding='unicode_escape')
    links = pd.read_excel(
        'C:\\Users\\Prabal\\Desktop\\Major2020\\webScrapping/Influencer_with_links.xlsx')
    # links=pd.read_excel('Influencer_with_links.xlsx')
    result = pd.merge(final, links, on="name")
    indexes = result[result['country'].str.lower() !=
                     countryName.lower()].index
    result.drop(indexes, inplace=True)
    result = result.sort_values(by="average_post_comments", ascending=False)
    onCountryFilter = []
    for d in result.values:
        onCountryFilter.append(
            {'name': d[0], 'category': d[10], 'id': d[11], 'src': d[12]})
        if(len(onCountryFilter) == 5):
            return onCountryFilter
    return onCountryFilter


def reduceVideoComment(countryName):
    final = pd.read_csv(
        'C:\\Users\\Prabal\\Desktop\\Major2020\\webScrapping/final_dataset.csv', encoding='unicode_escape')
    links = pd.read_excel(
        'C:\\Users\\Prabal\\Desktop\\Major2020\\webScrapping/Influencer_with_links.xlsx')
    # links=pd.read_excel('Influencer_with_links.xlsx')
    result = pd.merge(final, links, on="name")
    indexes = result[result['country'].str.lower() !=
                     countryName.lower()].index
    result.drop(indexes, inplace=True)
    result = result.sort_values(by="average_videos_comments", ascending=False)
    onCountryFilter = []
    for d in result.values:
        onCountryFilter.append(
            {'name': d[0], 'category': d[10], 'id': d[11], 'src': d[12]})
        if(len(onCountryFilter) == 5):
            return onCountryFilter
    return onCountryFilter


def reduceVideoViews(countryName):
    final = pd.read_csv(
        'C:\\Users\\Prabal\\Desktop\\Major2020\\webScrapping/final_dataset.csv', encoding='unicode_escape')
    links = pd.read_excel(
        'C:\\Users\\Prabal\\Desktop\\Major2020\\webScrapping/Influencer_with_links.xlsx')
    # links=pd.read_excel('Influencer_with_links.xlsx')
    result = pd.merge(final, links, on="name")
    indexes = result[result['country'].str.lower() !=
                     countryName.lower()].index
    result.drop(indexes, inplace=True)
    result = result.sort_values(by="average_videos_views", ascending=False)
    onCountryFilter = []
    for d in result.values:
        onCountryFilter.append(
            {'name': d[0], 'category': d[10], 'id': d[11], 'src': d[12]})
        if(len(onCountryFilter) == 5):
            return onCountryFilter
    return onCountryFilter


def countryInfluencer(request, countryName):
    onCountryFilter = reduceCountry(countryName)
    onCountryPostLike = reducePostLikes(countryName)
    onCountryVideocomments = reduceVideoComment(countryName)
    onCountryVideoViews = reduceVideoViews(countryName)
    print(onCountryVideocomments)
    return render(request, 'country.html', {'country': countryName, 'onCountryFilter': onCountryFilter, 'onCountryPostLike': onCountryPostLike, 'onCountryVideocomments': onCountryVideocomments, 'onCountryVideoViews': onCountryVideoViews})


def predict(request):
    if request.method == 'GET':
        return render(request, 'predict2.html')
    else:
        valueA = float(request.POST['valueA'])
        valueB = 1-valueA
        onBasisPost.clear()
        onBasisVideos.clear()
        onBasisCosine.clear()
        onBasisAll.clear()

        onBasisPostLinks.clear()
        onBasisVideosLinks.clear()
        onBasisCosineLinks.clear()
        onBasisAllLinks.clear()

        pearson_hypo_comment_likes_val_post.clear()
        pearson_hypo_likes_length_val_post.clear()
        pearson_hypo_comment_length_val_post.clear()
        t_hypo_comment_like_val_post.clear()
        t_hypo_like_length_val_post.clear()
        t_hypo_comment_length_val_post.clear()

        pearson_hypo_comment_likes_val_video.clear()
        pearson_hypo_likes_length_val_video.clear()
        pearson_hypo_comment_length_val_video.clear()
        t_hypo_comment_like_val_video.clear()
        t_hypo_like_length_val_video.clear()
        t_hypo_comment_length_val_video.clear()

        pearson_hypo_comment_likes_val_nlp.clear()
        pearson_hypo_likes_length_val_nlp.clear()
        pearson_hypo_comment_length_val_nlp.clear()
        t_hypo_comment_like_val_nlp.clear()
        t_hypo_like_length_val_nlp.clear()
        t_hypo_comment_length_val_nlp.clear()

        pearson_hypo_comment_likes_val_allto.clear()
        pearson_hypo_likes_length_val_allto.clear()
        pearson_hypo_comment_length_val_allto.clear()
        t_hypo_comment_like_val_allto.clear()
        t_hypo_like_length_val_allto.clear()
        t_hypo_comment_length_val_allto.clear()

        country = request.POST['countries']
        category = request.POST['categories']
        productDescription = request.POST['productDetail']
        hashtags = request.POST['hastags']
        print(country)
        print(category)
        print(productDescription)
        print(hashtags)
        data = redu(category, country)
        print(data)
        cos_data = pd.Series
        cos_data = cosineSimilairty(
            category, country, productDescription, hashtags)
        cos_data = cos_data.iloc[1:]
        cos_data = cos_data.reset_index(drop=True)
        data = data.reset_index(drop=True)
        data["cosine_sim"] = cos_data
        data.drop('description', inplace=True, axis=1)
        data.drop('tagged_names', inplace=True, axis=1)
        data.drop('country', inplace=True, axis=1)
        data.drop('category', inplace=True, axis=1)
        data = data.fillna(0)
        indices = pd.Series(data.name)
        data.drop('name', inplace=True, axis=1)
        data = normalize(data)
        data.insert(0, 'name', indices)
        # data=allto(data)
        # print(data)

        new_data = data.sort_values(by='cosine_sim', ascending=False)
        link_data = pd.read_excel(
            'C:\\Users\\Prabal\\Desktop\\Major2020\\webScrapping\\Influencer_with_links.xlsx')
        for d in new_data.values:
            if d[7] != 0:
                onBasisCosine.append({'name': d[0], 'value': d[7]})

                pearson_hypo_comment_likes_val_nlp.append(
                    pearson_hypo_comment_likes(d[0]))
                pearson_hypo_likes_length_val_nlp.append(
                    pearson_hypo_likes_length(d[0]))
                pearson_hypo_comment_length_val_nlp.append(
                    pearson_hypo_comment_length(d[0]))
                t_hypo_comment_like_val_nlp.append(t_hypo_comment_like(d[0]))
                t_hypo_like_length_val_nlp.append(t_hypo_like_length(d[0]))
                t_hypo_comment_length_val_nlp.append(
                    t_hypo_comment_length(d[0]))
                try:
                    val = link_data.set_index('name').to_dict()
                    dd = val['id'][d[0]]
                    print(d[0])
                    onBasisCosineLinks.append({'name': d[0], 'id': (dd)})
                except:
                    print('shashwat')

        data['acc_to_post'] = valueA*data['average_post_comments'] + \
            valueB*data['average_post_likes']
        new_data = data.sort_values(by='acc_to_post', ascending=False)
        for d in new_data.values:
            if d[8] != 0:
                onBasisPost.append({'name': d[0], 'value': d[8]})
                pearson_hypo_comment_likes_val_post.append(
                    pearson_hypo_comment_likes(d[0]))
                pearson_hypo_likes_length_val_post.append(
                    pearson_hypo_likes_length(d[0]))
                pearson_hypo_comment_length_val_post.append(
                    pearson_hypo_comment_length(d[0]))
                t_hypo_comment_like_val_post.append(t_hypo_comment_like(d[0]))
                t_hypo_like_length_val_post.append(t_hypo_like_length(d[0]))
                t_hypo_comment_length_val_post.append(
                    t_hypo_comment_length(d[0]))
                try:
                    val = link_data.set_index('name').to_dict()
                    dd = val['id'][d[0]]
                    onBasisPostLinks.append({'name': d[0], 'id': (dd)})
                except:
                    print("error in "+d[0])

        data['acc_to_videos'] = valueA*data['average_videos_comments'] + \
            valueB*data['average_videos_views']
        new_data = data.sort_values(by='acc_to_videos', ascending=False)
        for d in new_data.values:
            if d[9] != 0:
                onBasisVideos.append({'name': d[0], 'value': d[9]})
                pearson_hypo_comment_likes_val_video.append(
                    pearson_hypo_comment_likes(d[0]))
                pearson_hypo_likes_length_val_video.append(
                    pearson_hypo_likes_length(d[0]))
                pearson_hypo_comment_length_val_video.append(
                    pearson_hypo_comment_length(d[0]))
                t_hypo_comment_like_val_video.append(t_hypo_comment_like(d[0]))
                t_hypo_like_length_val_video.append(t_hypo_like_length(d[0]))
                t_hypo_comment_length_val_video.append(
                    t_hypo_comment_length(d[0]))
                try:
                    val = link_data.set_index('name').to_dict()
                    dd = val['id'][d[0]]
                    onBasisVideosLinks.append({'name': d[0], 'id': (dd)})
                except:
                    pass

        new_data = allto(data)
        new_data = new_data.sort_values(by='cumulative', ascending=False)
        for d in new_data.values:
            if d[10] != 0:
                onBasisAll.append({'name': d[0], 'value': d[11-1]})
                pearson_hypo_comment_likes_val_allto.append(
                    pearson_hypo_comment_likes(d[0]))
                pearson_hypo_likes_length_val_allto.append(
                    pearson_hypo_likes_length(d[0]))
                pearson_hypo_comment_length_val_allto.append(
                    pearson_hypo_comment_length(d[0]))
                t_hypo_comment_like_val_allto.append(t_hypo_comment_like(d[0]))
                t_hypo_like_length_val_allto.append(t_hypo_like_length(d[0]))
                t_hypo_comment_length_val_allto.append(
                    t_hypo_comment_length(d[0]))
                try:
                    val = link_data.set_index('name').to_dict()
                    dd = val['id'][d[0]]
                    onBasisAllLinks.append({'name': d[0], 'id': (dd)})
                except:
                    pass
        return render(request, 'output.html', {'onBasisAllLinks': onBasisAllLinks[0:6], 'onBasisCosineLinks': onBasisCosineLinks[0:6], 'onBasisPostLinks': onBasisPostLinks[0:6], 'onBasisVideosLinks': onBasisVideosLinks[0:6], 'pearson_hypo_comment_length_val_post': pearson_hypo_comment_length_val_post[0:6], 'pearson_hypo_comment_likes_val_post': pearson_hypo_comment_likes_val_post[0:6], 'pearson_hypo_likes_length_val_post': pearson_hypo_likes_length_val_post[0:6], 't_hypo_comment_length_val_post': t_hypo_comment_length_val_post[0:6], 't_hypo_comment_like_val_post': t_hypo_comment_like_val_post[0:6], 't_hypo_like_length_val_post': t_hypo_like_length_val_post[0:6], 'pearson_hypo_comment_length_val_video': pearson_hypo_comment_length_val_video[0:6],        'pearson_hypo_comment_likes_val_video': pearson_hypo_comment_likes_val_video[0:6],        'pearson_hypo_likes_length_val_video': pearson_hypo_likes_length_val_video[0:6],        't_hypo_comment_length_val_video': t_hypo_comment_length_val_video[0:6],        't_hypo_comment_like_val_video': t_hypo_comment_like_val_video[0:6],        't_hypo_like_length_val_video': t_hypo_like_length_val_video[0:6],        'pearson_hypo_comment_length_val_nlp': pearson_hypo_comment_length_val_nlp[0:6],        'pearson_hypo_comment_likes_val_nlp': pearson_hypo_comment_likes_val_nlp[0:6],        'pearson_hypo_likes_length_val_nlp': pearson_hypo_likes_length_val_nlp[0:6],        't_hypo_comment_length_val_nlp': t_hypo_comment_length_val_nlp[0:6],        't_hypo_comment_like_val_nlp': t_hypo_comment_like_val_nlp[0:6],        't_hypo_like_length_val_nlp': t_hypo_like_length_val_nlp[0:6],                'pearson_hypo_comment_length_val_allto': pearson_hypo_comment_length_val_allto[0:6],        'pearson_hypo_comment_likes_val_allto': pearson_hypo_comment_likes_val_allto[0:6],        'pearson_hypo_likes_length_val_allto': pearson_hypo_likes_length_val_allto[0:6],        't_hypo_comment_length_val_allto': t_hypo_comment_length_val_allto[0:6],     't_hypo_comment_like_val_allto': t_hypo_comment_like_val_allto[0:6], 't_hypo_like_length_val_allto': t_hypo_like_length_val_allto[0:6]})


def final_output(request):
    print(pearson_hypo_comment_length_val_post[0:6])
    print(pearson_hypo_comment_likes_val_post[0:6])
    print(pearson_hypo_likes_length_val_post[0:6])
    print(t_hypo_comment_length_val_post[0:6])
    print(t_hypo_comment_like_val_post[0:6])
    print(t_hypo_like_length_val_post[0:6])
    print("value changed its noe video")
    print(pearson_hypo_comment_length_val_video[0:6])
    print(pearson_hypo_comment_likes_val_video[0:6])
    print(pearson_hypo_likes_length_val_video[0:6])
    print(t_hypo_comment_length_val_video[0:6])
    print(t_hypo_comment_like_val_video[0:6])
    print(t_hypo_like_length_val_video[0:6])
    print("value changed its now nlp")
    print(pearson_hypo_comment_length_val_nlp[0:6])
    print(pearson_hypo_comment_likes_val_nlp[0:6])
    print(pearson_hypo_likes_length_val_nlp[0:6])
    print(t_hypo_comment_length_val_nlp[0:6])
    print(t_hypo_comment_like_val_nlp[0:6])
    print(t_hypo_like_length_val_nlp[0:6])
    print("value changed its now alltogether")
    print(pearson_hypo_comment_length_val_allto[0:6])
    print(pearson_hypo_comment_likes_val_allto[0:6])
    print(pearson_hypo_likes_length_val_allto[0:6])
    print(t_hypo_comment_length_val_allto[0:6])
    print(t_hypo_comment_like_val_allto[0:6])
    print(t_hypo_like_length_val_allto[0:6])

    data = {
        "post": onBasisPost[0:6],
        "video": onBasisVideos[0:6],
        "cosine": onBasisCosine[0:6],
        "allto": onBasisAll[0:6]
    }
    return JsonResponse(data)
