import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pycountry
from PIL import Image 
from streamlit_option_menu import option_menu
from streamlit_tags import st_tags_sidebar
from streamlit_drawable_canvas import st_canvas
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from scipy import spatial
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder

import plotly.graph_objects as go

icon_img = Image.open("icon.png")
st.set_page_config(page_title="Web-based recommender systems", page_icon=icon_img)

st.sidebar.markdown("<h1 style='text-align: center; color: black;'>Diploma Thesis</h1>", unsafe_allow_html=True)

with st.sidebar:
    selected = option_menu("Menu", ["Home","Recommend me"],icons=['house-fill', 'person-lines-fill'],
                           menu_icon="mortarboard-fill", default_index=0, orientation="vertical")

@st.cache(allow_output_mutation=True,show_spinner=False)
def load_data():
    all_players = pd.read_excel("Big5_EuropeanLeagues_2021_2022.xlsx")
    
    all_players['Player'] = all_players['Player'].str.split('\\').str[0]
    all_players['Nation'] = all_players['Nation'].str.split(' ').str[1]
    all_players['Age'] = all_players['Age'].str.split('-').str[0] 

    all_players["Age"] = all_players["Age"].astype("float")
    all_players["Age"] = all_players["Age"].astype("Int64")

    all_players["Comp"] = all_players["Comp"].apply(lambda x: x.replace("fr Ligue 1", "Ligue 1"))
    all_players["Comp"] = all_players["Comp"].apply(lambda x: x.replace("eng Premier League", "Premier League"))
    all_players["Comp"] = all_players["Comp"].apply(lambda x: x.replace("de Bundesliga", "Bundesliga"))
    all_players["Comp"] = all_players["Comp"].apply(lambda x: x.replace("es La Liga", "La Liga"))
    all_players["Comp"] = all_players["Comp"].apply(lambda x: x.replace("it Serie A", "Serie A"))


    all_players["Nation"] = all_players["Nation"].apply(lambda x: x.replace("ALG", "DZA"))
    all_players["Nation"] = all_players["Nation"].apply(lambda x: x.replace("ANG", "AGO"))
    all_players["Nation"] = all_players["Nation"].apply(lambda x: x.replace("BUL", "BGR"))
    all_players["Nation"] = all_players["Nation"].apply(lambda x: x.replace("CGO", "COG"))
    all_players["Nation"] = all_players["Nation"].apply(lambda x: x.replace("CHA", "TCD"))
    all_players["Nation"] = all_players["Nation"].apply(lambda x: x.replace("CHI", "CHL"))
    all_players["Nation"] = all_players["Nation"].apply(lambda x: x.replace("CRC", "CRI"))
    all_players["Nation"] = all_players["Nation"].apply(lambda x: x.replace("CRO", "HRV"))
    all_players["Nation"] = all_players["Nation"].apply(lambda x: x.replace("CTA", "CAF"))
    all_players["Nation"] = all_players["Nation"].apply(lambda x: x.replace("DEN", "DNK"))
    all_players["Nation"] = all_players["Nation"].apply(lambda x: x.replace("EQG", "GNQ"))
    all_players["Nation"] = all_players["Nation"].apply(lambda x: x.replace("GAM", "GMB"))
    all_players["Nation"] = all_players["Nation"].apply(lambda x: x.replace("GER", "DEU"))
    all_players["Nation"] = all_players["Nation"].apply(lambda x: x.replace("GRE", "GRC"))
    all_players["Nation"] = all_players["Nation"].apply(lambda x: x.replace("GRN", "GRD"))
    all_players["Nation"] = all_players["Nation"].apply(lambda x: x.replace("GUI", "GIN"))
    all_players["Nation"] = all_players["Nation"].apply(lambda x: x.replace("HON", "HND"))
    all_players["Nation"] = all_players["Nation"].apply(lambda x: x.replace("MAD", "MDG"))
    all_players["Nation"] = all_players["Nation"].apply(lambda x: x.replace("MTN", "MRT"))
    all_players["Nation"] = all_players["Nation"].apply(lambda x: x.replace("NED", "NLD"))
    all_players["Nation"] = all_players["Nation"].apply(lambda x: x.replace("PAR", "PRY"))
    all_players["Nation"] = all_players["Nation"].apply(lambda x: x.replace("PHI", "PHL"))
    all_players["Nation"] = all_players["Nation"].apply(lambda x: x.replace("POR", "PRT"))
    all_players["Nation"] = all_players["Nation"].apply(lambda x: x.replace("RSA", "ZAF"))
    all_players["Nation"] = all_players["Nation"].apply(lambda x: x.replace("SKN", "KNA"))
    all_players["Nation"] = all_players["Nation"].apply(lambda x: x.replace("SUI", "CHE"))
    all_players["Nation"] = all_players["Nation"].apply(lambda x: x.replace("TOG", "TGO"))
    all_players["Nation"] = all_players["Nation"].apply(lambda x: x.replace("URU", "URY"))
    all_players["Nation"] = all_players["Nation"].apply(lambda x: x.replace("ZIM", "ZWE"))
    all_players["Nation"] = all_players["Nation"].apply(lambda x: x.replace("ZAM", "ZMB"))


    countries = []
    for i in range(len(all_players.index)):
        if(all_players.iloc[i,1]=="ENG"):
            name = "England"
        elif(all_players.iloc[i,1]=="NIR"):
            name = "Northern Ireland"
        elif(all_players.iloc[i,1]=="SCO"):
            name = "Scotland"
        elif(all_players.iloc[i,1]=="WAL"):
            name = "Wales"

        elif(all_players.iloc[i,1]=="KVX"):
            name = "Kosovo"
        else:
            name = pycountry.countries.get(alpha_3=all_players.iloc[i,1]).name

        countries.append(name)


    all_players.insert(2, "Nationality", countries)
    
    all_players['Pos'] = all_players['Pos'].str[:2]
    
    all_players["Position"] = all_players["Pos"].map({'GK':1,'DF':2,'MF':3,'FW':4})
    
    return all_players


all_players = load_data()


if selected == "Home":
    st.markdown("<h1 style='text-align: center; color: black;'>Top 5 European leagues players stats</h1>", unsafe_allow_html=True)
    gb = GridOptionsBuilder.from_dataframe(all_players)
    gb.configure_pagination()
    gridOptions = gb.build()
    AgGrid(all_players, gridOptions=gridOptions)

if selected == "Recommend me":
    with st.sidebar.expander("Recommender System",expanded=True):
        numberOfPlayers = st.number_input('Number of players to display',value=3,min_value=1,max_value=5)
        myTeam = st.selectbox("Choose your team", all_players["Squad"].unique().tolist(),index=22)
        df_myTeam= all_players.loc[all_players['Squad'] == myTeam]
        df_myTeam.reset_index(inplace = True,drop = True)
        injuredPlayer = st.selectbox("Injured player", df_myTeam["Player"],index=1)
        fromWhere = st.radio("Recommendations",('Only from player position','From all position',),index=0)
        
        options = df_myTeam.values[:,:]
       
    with st.expander("My team - "+myTeam,expanded=False):
       # st.write("My team -", myTeam)    
        AgGrid(df_myTeam)
    
    statistics = df_myTeam.iloc[:, 6:]
    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(statistics), columns=statistics.columns)
    n = df_myTeam.Player.count()
    recommendations = NearestNeighbors(n_neighbors=n)
    recommendations.fit(X)
    player_index = recommendations.kneighbors(X,return_distance=False)
    
    players = []
    cosine_similarity = []
    values = []
   
    def get_index(x):
        return df_myTeam[df_myTeam['Player']==x].index.tolist()[0]

    def get_position(x):
        return df_myTeam[df_myTeam['Player']==x].Pos.tolist()[0]

    def recommend_similar_player_like(player):
        players.clear()
        cosine_similarity.clear()
        values.clear()
        index=  get_index(player)
        position=  get_position(player)
        for i in player_index[index][1:]:
            if(fromWhere == "Only from player position" and df_myTeam.iloc[i]['Pos']==position):
                players.append(df_myTeam.iloc[i]['Player'])
                cosine_similarity.append(np.round((1 - spatial.distance.cosine(X.iloc[index], X.iloc[i]))*100,2))
                values.append(df_myTeam.loc[i, df_myTeam.columns != 'Player'])
            elif(fromWhere == "From all position"):
                players.append(df_myTeam.iloc[i]['Player'])
                cosine_similarity.append(np.round((1 - spatial.distance.cosine(X.iloc[index], X.iloc[i]))*100,2))
                values.append(df_myTeam.loc[i, df_myTeam.columns != 'Player'])
                
        recommended_players_df = pd.DataFrame(values, columns=df_myTeam.columns.values[1:])
        recommended_players_df.insert(0, "Player", players)
        recommended_players_df.insert(1, "Similarity %", cosine_similarity)
        recommended_players_df = recommended_players_df.sort_values(by=['Similarity %'], ascending=False)
        recommended_players_df.reset_index(inplace = True,drop = True)
        
        st.write("Similar football players as", injuredPlayer+":")
        
        return recommended_players_df.head(numberOfPlayers)
    
    df_RecommendedPlayers = recommend_similar_player_like(injuredPlayer)
    AgGrid(df_RecommendedPlayers)
    
    aaa = pd.DataFrame(scaler.fit_transform(df_RecommendedPlayers.iloc[:, 9:30]), columns=df_RecommendedPlayers.columns[9:30])
    
    fig = go.Figure()

    for i in range(len(df_RecommendedPlayers.index)):
        fig.add_trace(
                    go.Scatterpolar(
                                    r=aaa.loc[i].values,
                                    theta=aaa.columns,
                                    fill='toself',
                                    #fillcolor = 'aliceblue',
                                    name=df_RecommendedPlayers["Player"].loc[i]+
                                    " (Similarity "+str(df_RecommendedPlayers["Similarity %"].loc[i]) +" %)",
                                    showlegend=True,
                                    )
                    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                            visible=True,
                             range=[-0.1, 1.1]
                        )
                ),

        title="Similar players to "+ injuredPlayer
    )
    st.plotly_chart(fig, use_container_width=True)