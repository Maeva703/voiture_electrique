
"""
DATAExam.py

Streamlit app for electric vehicle market analysis
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
import numpy as np

# Set the page config
st.set_page_config(page_title="Analyse du Marché des Véhicules Électriques", layout="wide")

# Load data
@st.cache
def load_data():
    return pd.read_csv("Electric_Vehicle_Population_Data.csv")

ev_data = load_data()

st.title('Analyse de la Taille du Marché des Véhicules Électriques')
st.markdown("""
Cette application vous permet d'analyser le marché des véhicules électriques aux États-Unis. Utilisez les options du menu pour explorer les données de différentes manières.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
menu = st.sidebar.selectbox("Choisissez une analyse", ["Introduction", "Données brutes", "Adoption au fil du temps", "Répartition géographique", "Types de véhicules", "Marques populaires", "Modèles populaires", "Autonomie électrique", "Estimation du marché"])

# Introduction
if menu == "Introduction":
    st.subheader("Introduction")
    st.markdown("""
    Cette application interactive vous permet d'explorer et d'analyser le marché des véhicules électriques aux États-Unis. Vous pouvez visualiser l'adoption au fil du temps, la répartition géographique, les types de véhicules, les marques et modèles populaires, ainsi que des estimations du marché futur.
    """)

# Show dataset
if menu == "Données brutes":
    st.subheader("Données brutes")
    st.write(ev_data)

# Data cleaning
ev_data = ev_data.dropna()

# Year-wise adoption
if menu == "Adoption au fil du temps":
    st.subheader("Adoption de véhicules électriques au fil du temps")
    plt.figure(figsize=(12,6))
    ev_adoption_by_year = ev_data['Model Year'].value_counts().sort_index()
    sns.barplot(x=ev_adoption_by_year.index, y=ev_adoption_by_year.values, palette="viridis")
    plt.title("ADOPTION EV AU FIL DU TEMPS")
    plt.xlabel("Année modèle")
    plt.ylabel("Nombres de véhicules immatriculés")
    plt.xticks(rotation=45)
    st.pyplot(plt)

# Top counties and city distribution
if menu == "Répartition géographique":
    st.subheader("Répartition géographique des principaux comtés")
    ev_county_distribution = ev_data['County'].value_counts()
    top_counties = ev_county_distribution.head(3).index
    top_counties_data = ev_data[ev_data['County'].isin(top_counties)]
    ev_city_distribution_top_counties = top_counties_data.groupby(['County', 'City']).size().sort_values(ascending=False).reset_index(name='Number of Vehicles')
    top_cities = ev_city_distribution_top_counties.head(10)
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Number of Vehicles', y='City', hue='County', data=top_cities, palette="magma")
    plt.title('Meilleurs villes des principaux comtés par inscription de véhicules')
    plt.xlabel('Nombre de véhicules enregistrés')
    plt.ylabel('City')
    plt.legend(title='County')
    st.pyplot(plt)

# EV type distribution
if menu == "Types de véhicules":
    st.subheader("Distribution des types de véhicules électriques")
    ev_type_distribution = ev_data['Electric Vehicle Type'].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=ev_type_distribution.values, y=ev_type_distribution.index, palette="rocket")
    plt.title('Distribution des types de véhicules électriques')
    plt.xlabel('Nombres de véhicules immatriculés')
    plt.ylabel('Type de véhicules électrique')
    st.pyplot(plt)

# Top EV makes
if menu == "Marques populaires":
    st.subheader("Top 10 des marques électriques les plus populaires")
    ev_make_distribution = ev_data['Make'].value_counts().head(10)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=ev_make_distribution.values, y=ev_make_distribution.index, palette="cubehelix")
    plt.title('Top 10 des marques électriques les plus populaires')
    plt.xlabel('Nombres de véhicules immatriculés')
    plt.ylabel('Make')
    st.pyplot(plt)

# Top EV models by top makes
if menu == "Modèles populaires":
    st.subheader("Meilleurs modèles dans le top 3 des marques par inscriptions de véhicule")
    top_3_makes = ev_make_distribution.head(3).index
    top_makes_data = ev_data[ev_data['Make'].isin(top_3_makes)]
    ev_model_distribution_top_makes = top_makes_data.groupby(['Make', 'Model']).size().sort_values(ascending=False).reset_index(name='Number of Vehicles')
    top_models = ev_model_distribution_top_makes.head(10)
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Number of Vehicles', y='Model', hue='Make', data=top_models, palette="viridis")
    plt.title('Meilleurs modèles dans le top 3 des marques par inscriptions de véhicule')
    plt.xlabel('Nombre de véhicules immatriculés')
    plt.ylabel('Modèles')
    plt.legend(title='Make', loc='center right')
    st.pyplot(plt)

# Electric range distribution
if menu == "Autonomie électrique":
    st.subheader("Distribution de l'autonomie électrique")
    plt.figure(figsize=(12, 6))
    sns.histplot(ev_data['Electric Range'], bins=30, kde=True, color='royalblue')
    plt.title('Distribution de gammes de véhicules électriques')
    plt.xlabel('Autonomie électriques (miles)')
    plt.ylabel('Nombre de véhicules')
    plt.axvline(ev_data['Electric Range'].mean(), color='red', linestyle='--', label=f'Mean Range: {ev_data["Electric Range"].mean():.2f} miles')
    plt.legend()
    st.pyplot(plt)

# Average range by model year
if menu == "Autonomie électrique moyenne par année modèle":
    st.subheader("Autonomie électrique moyenne par année modèle")
    average_range_by_year = ev_data.groupby('Model Year')['Electric Range'].mean().reset_index()
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Model Year', y='Electric Range', data=average_range_by_year, marker='o', color='green')
    plt.title('Autonomie électrique moyenne par année modèle')
    plt.xlabel('Model Year')
    plt.ylabel('Autonomie électrique moyenne (miles)')
    plt.grid(True)
    st.pyplot(plt)

# Market size estimation
if menu == "Estimation du marché":
    st.subheader("Estimation de la taille du marché des véhicules électriques aux États-Unis")
    ev_registration_counts = ev_data['Model Year'].value_counts().sort_index()
    filtered_years = ev_registration_counts[ev_registration_counts.index <= 2023]

    def exp_growth(x, a, b):
        return a * np.exp(b * x)

    x_data = filtered_years.index - filtered_years.index.min()
    y_data = filtered_years.values
    params, covariance = curve_fit(exp_growth, x_data, y_data)
    forecast_years = np.arange(2024, 2024 + 6) - filtered_years.index.min()
    forecasted_values = exp_growth(forecast_years, *params)
    forecasted_evs = dict(zip(forecast_years + filtered_years.index.min(), forecasted_values))

    years = np.arange(filtered_years.index.min(), 2029 + 1)
    actual_years = filtered_years.index
    forecast_years_full = np.arange(2024, 2029 + 1)
    actual_values = filtered_years.values
    forecasted_values_full = [forecasted_evs[year] for year in forecast_years_full]

    plt.figure(figsize=(12, 8))
    plt.plot(actual_years, actual_values, 'bo-', label='Actual Registrations')
    plt.plot(forecast_years_full, forecasted_values_full, 'ro--', label='Forecasted Registrations')
    plt.title('Marché actuel et estimé des véhicules électriques')
    plt.xlabel('Year')
    plt.ylabel("Nombres d'immatriculation EV")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

st.sidebar.markdown(""" **Conclusion**
Ainsi, l’analyse de la taille du marché est un aspect crucial de l’étude de marché qui détermine le volume des ventes potentielles sur""")