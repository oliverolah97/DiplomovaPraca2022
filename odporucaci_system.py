import streamlit as st
import numpy as np
import pandas as pd
from st_aggrid import AgGrid
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu
from streamlit_tags import st_tags_sidebar
from PIL import Image 
import pycountry
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from scipy import spatial
from streamlit_drawable_canvas import st_canvas

icon_img = Image.open("icon.png")

st.set_page_config(
    page_title="Webové odporúčacie systémy",
    page_icon=icon_img
)

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

st.sidebar.markdown("<h1 style='text-align: center; color: black;'>Diplomová práca</h1>", unsafe_allow_html=True)

with st.sidebar:
    selected = option_menu("Menu", ["Home","Recommend me","Draw"],icons=['house-fill', 'person-lines-fill','palette-fill'],
                           menu_icon="mortarboard-fill", default_index=0, orientation="vertical")


if selected == "Home":
    video_file = open('videoFile.mp4', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)

    
    
if selected == "Recommend me":
    #AgGrid(all_players)
    with st.sidebar.expander("Recommender System",expanded=True):
        numberOfPlayers = st.number_input('Number of players to display',value=5,min_value=0,max_value=15)
        myTeam = st.selectbox("Choose your team", all_players["Squad"].unique().tolist(),index=22)
        df_myTeam= all_players.loc[all_players['Squad'] == myTeam]
        injuredPlayer = st.selectbox("Injured player", df_myTeam["Player"],index=2)
      
        
        options = df_myTeam.values[:,:].tolist()
        #selected = st.multiselect('Choose your lineup11',options,help="ssdsdsdsdsd")   
   
        
        # Top 5 strelcov za sezonu 2020/2021
        gls = df_myTeam.sort_values(by = 'Gls',ascending = False).head(5)
        gls = gls[['Player','Gls']]
        x = gls['Player']
        y = gls['Gls']
        plt.figure(figsize=(3,3))
        ax= sns.barplot(x=y, y=x, palette = 'dark', orient='h')
        plt.xticks()
        plt.xlabel('Number of goals', size = 10, color="k") 
        plt.ylabel('Players', size =10 ) 
        plt.title('Top 5 scorers - '+ myTeam,size=10)

        for index, value in enumerate(y):
            plt.text(value, index, str(value))

        st.sidebar.pyplot(plt)
       
   
        
    AgGrid(df_myTeam)
    
    statistics = all_players.iloc[:, 6:]
    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(statistics), columns=statistics.columns)
    recommendations = NearestNeighbors(n_neighbors=50)
    recommendations.fit(X)
    player_index = recommendations.kneighbors(X,return_distance=False)
    
    players = []
    cosine_similarity = []
    values = []
    def get_index(x):
        return all_players[all_players['Player']==x].index.tolist()[0]

    def get_position(x):
        return all_players[all_players['Player']==x].Pos.tolist()[0]

    def recommend_similar_player_like(player):
        players.clear()
        cosine_similarity.clear()
        values.clear()
        index=  get_index(player)
        position=  get_position(player)
        for i in player_index[index][1:]:
            if(all_players.iloc[i]['Pos']==position):
                players.append(all_players.iloc[i]['Player'])
                cosine_similarity.append(np.round((1 - spatial.distance.cosine(statistics.iloc[index], statistics.iloc[i]))*100,2))
                values.append(all_players.loc[i, all_players.columns != 'Player'])
        recommended_players_df = pd.DataFrame(values, columns=all_players.columns.values[1:])
        recommended_players_df.insert(0, "Player", players)
        recommended_players_df.insert(1, "Similarity %", cosine_similarity)
        recommended_players_df = recommended_players_df.sort_values(by=['Similarity %'], ascending=False)
        recommended_players_df.reset_index(inplace = True,drop = True)
        
       
        st.write("Similar football players as ", injuredPlayer,":")
        
        return AgGrid(recommended_players_df.head(numberOfPlayers))
    
    
    recommend_similar_player_like(injuredPlayer)
        
   
    #AgGrid(recommended_players_df)
   
    #myLineup = pd.DataFrame.from_records(selected)
    
    #if myLineup.empty:
     #   st.write("LineUp empty")
    #else:
     #   myLineup.columns = df_myTeam.columns.tolist()
      #  AgGrid(myLineup)

if selected == "Draw":
    # Specify canvas parameters in application
    color = st.sidebar.color_picker("Color: ")
    bg_lineup = st.sidebar.radio(
         "Select your lineup",
         ('','4-4-2', '4-3-3'),index=2)
    
    bg_image = ""
    if bg_lineup == '4-4-2':
            bg_image = Image.open("lineup4_4_2.png")

    elif bg_lineup == "4-3-3":
            bg_image = Image.open("lineup4_3_3.png")
  
    # Create a canvas component
    canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=2,
    stroke_color=color,
    background_color = "#eee",
    background_image=bg_image if bg_image else None,
    update_streamlit=False,
    height=400,
    width=300,
    drawing_mode="freedraw",
    key="canvas")
    
    


