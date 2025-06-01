import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
from scipy import stats
import numpy as np

df = pd.read_csv("marketing_campaign.csv", sep='\t')
df['Age'] = 2025 - df['Year_Birth']
df['Children'] = df['Kidhome'] + df['Teenhome']
df['Total_Spending'] = df[['MntWines', 'MntFruits', 'MntMeatProducts',
                           'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum(axis=1)
df = df.dropna(subset=['Income'])
df['AgeGroup'] = pd.cut(df['Age'], bins=[17, 25, 35, 50, 70, 100],
                        labels=['18-25', '26-35', '36-50', '51-70', '70+'])
df['Campaign_Response'] = df[['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3',
                              'AcceptedCmp4', 'AcceptedCmp5', 'Response']].sum(axis=1)
df['Participated'] = df['Campaign_Response'] > 0

# Гипотезы с t-тестами и ANOVA
with_kids = df[df['Children'] > 0]['MntSweetProducts']
without_kids = df[df['Children'] == 0]['MntSweetProducts']
t1_stat, t1_p = stats.ttest_ind(with_kids, without_kids, equal_var=False)

groups = [group['Total_Spending'].values for _, group in df.groupby('Education')]
f2_stat, f2_p = stats.f_oneway(*groups)

divorced = df[df['Marital_Status'] == 'Divorced']['MntWines']
married = df[df['Marital_Status'] == 'Married']['MntWines']
t3_stat, t3_p = stats.ttest_ind(divorced, married, equal_var=False)

part = df[df['Participated']]['Total_Spending']
non_part = df[~df['Participated']]['Total_Spending']
t4_stat, t4_p = stats.ttest_ind(part, non_part, equal_var=False)

complain_yes = df[df['Complain'] == 1]['Total_Spending']
complain_no = df[df['Complain'] == 0]['Total_Spending']
t5_stat, t5_p = stats.ttest_ind(complain_yes, complain_no, equal_var=False)

# Дополнительные гипотезы и математический анализ
r_income_web, p_income_web = stats.pearsonr(df['Income'], df['NumWebPurchases'])
r_age_web, p_age_web = stats.pearsonr(df['Age'], df['NumWebPurchases'])
r_kids_spend, p_kids_spend = stats.pearsonr(df['Children'], df['Total_Spending'])

# Регрессионный анализ: доход ~ онлайн-покупки
slope, intercept, r_val, p_val, std_err = stats.linregress(df['Income'], df['NumWebPurchases'])

print("\n=== Статистический анализ гипотез ===")
print(f"1. Дети и сладости: t = {t1_stat:.2f}, p = {t1_p:.3e}, Значимость: {t1_p < 0.05}")
print(f"2. Образование и траты: F = {f2_stat:.2f}, p = {f2_p:.3e}, Значимость: {f2_p < 0.05}")
print(f"3. Marital (Divorced vs Married) и вино: t = {t3_stat:.2f}, p = {t3_p:.3e}, Значимость: {t3_p < 0.05}")
print(f"4. Кампании и траты: t = {t4_stat:.2f}, p = {t4_p:.3e}, Значимость: {t4_p < 0.05}")
print(f"5. Жалобы и траты: t = {t5_stat:.2f}, p = {t5_p:.3e}, Значимость: {t5_p < 0.05}")
print(f"6. Доход и онлайн-покупки: r = {r_income_web:.2f}, p = {p_income_web:.3e}, Значимость: {p_income_web < 0.05}")
print(f"7. Возраст и онлайн-покупки: r = {r_age_web:.2f}, p = {p_age_web:.3e}, Значимость: {p_age_web < 0.05}")
print(f"8. Кол-во детей и траты: r = {r_kids_spend:.2f}, p = {p_kids_spend:.3e}, Значимость: {p_kids_spend < 0.05}")
print("\n=== Регрессионная модель: Онлайн-покупки = a * Доход + b ===")
print(f"slope = {slope:.4f}, intercept = {intercept:.2f}, r^2 = {r_val**2:.3f}, p = {p_val:.3e}")

# Построение графиков для дашборда
fig_age_dist = px.histogram(df, x='Age', nbins=20, title='Распределение возраста клиентов')
age_segment_share = len(df[(df['Age'] >= 30) & (df['Age'] <= 50)]) / len(df) * 100
fig_age_segment = go.Figure(go.Indicator(
    mode="gauge+number",
    value=age_segment_share,
    title={'text': "% клиентов 30-50 лет"},
    gauge={'axis': {'range': [0, 100]}}
))
fig_sweets_kids = px.bar(
    df.groupby('Children')['MntSweetProducts'].mean().reset_index(),
    x='Children', y='MntSweetProducts',
    title='Средние траты на сладости по количеству детей'
)
fig_web_age = px.bar(
    df.groupby('AgeGroup')['NumWebPurchases'].mean().reset_index(),
    x='AgeGroup', y='NumWebPurchases',
    title='Онлайн-покупки по возрастным группам'
)
fig_edu_spending = px.box(
    df, x='Education', y='Total_Spending',
    title='Общие траты по уровням образования'
)
fig_wine_marital = px.box(
    df, x='Marital_Status', y='MntWines',
    title='Покупка вина по семейному статусу'
)
fig_complain_spend = px.box(
    df, x='Complain', y='Total_Spending',
    title='Жалобы и общие траты',
    labels={'Complain': 'Жалоба (0 - нет, 1 - да)'}
)
fig_campaign_spend = px.box(
    df, x='Participated', y='Total_Spending',
    title='Участие в кампаниях и траты',
    labels={'Participated': 'Участвовал'}
)
fig_deals_spend = px.scatter(
    df, x='NumDealsPurchases', y='Total_Spending',
    title='Покупки по акциям и общие траты'
)
fig_income_web = px.scatter(
    df, x='Income', y='NumWebPurchases',
    title='Доход и онлайн-покупки'
)

# Интерфейс Dash
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = dbc.Container([
    html.H1("Аналитика по покупательским гипотезам", className="my-4"),
    dbc.Row([
        dbc.Col(dcc.Graph(figure=fig_age_dist), md=6),
        dbc.Col(dcc.Graph(figure=fig_age_segment), md=6),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(figure=fig_sweets_kids), md=6),
        dbc.Col(dcc.Graph(figure=fig_web_age), md=6),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(figure=fig_edu_spending), md=12),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(figure=fig_wine_marital), md=6),
        dbc.Col(dcc.Graph(figure=fig_complain_spend), md=6),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(figure=fig_campaign_spend), md=6),
        dbc.Col(dcc.Graph(figure=fig_deals_spend), md=6),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(figure=fig_income_web), md=12),
    ])
], fluid=True)

if __name__ == "__main__":
    app.run(debug=True)
