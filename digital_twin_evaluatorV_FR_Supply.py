# Import required libraries
import streamlit as st
import pandas as pd
import numpy as np
import os
import dotenv
import plotly.graph_objects as go
from functools import lru_cache
from streamlit_gsheets import GSheetsConnection
import uuid
import json
from streamlit_scroll_to_top import scroll_to_here
from openai import OpenAI

# Ensure session state for scrolling exists
if "scroll_to_top" not in st.session_state:
    st.session_state.scroll_to_top = False

if "scroll_to_bottom" not in st.session_state:
    st.session_state.scroll_to_bottom = False

# Initialize the OpenAI client
dotenv.load_dotenv()

openai_api_key = st.secrets["openai"]["openai_api_key"]
client = OpenAI(api_key=openai_api_key)

#os.getenv("OPENAI_API_KEY")
from openai import OpenAI  # Import the client
from openai import OpenAIError  # Import the updated exception handling
import time
from PIL import Image
from audio_recorder_streamlit import audio_recorder
import base64
from io import BytesIO

# Initialize the OpenAI client
dotenv.load_dotenv()
#openai.api_key = st.secrets["openai"]["openai_api_key"]

# Load article data from file
with open(".streamlit/DT_info.json", "r", encoding="utf-8") as f:
    articles_data = json.load(f)  # a list of dicts

# Set the embedding model
EMBEDDING_MODEL = "text-embedding-ada-002"

@st.cache_resource
def get_embedding(text):
    """
    Returns a normalized embedding vector for `text` using OpenAI.
    """
    if not text:
        raise ValueError("No text provided for embedding.")
    
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=[text])
    embedding_vector = response.data[0].embedding  
    return embedding_vector / np.linalg.norm(embedding_vector)  # Normalize

with st.spinner("Creating embeddings for each article summary. This may take a while..."):
    placeholder = st.empty()  # Create a placeholder for animation

    # Build article corpus with embeddings
    article_corpus = []
    for article in articles_data:
        title = article.get("title", "")
        summary = article.get("summary", "")
        text_to_embed = summary
        embedding_vec = get_embedding(text_to_embed)
        article_corpus.append({
            "title": title,
            "summary": summary,
            "embedding": embedding_vec
        })

    placeholder.empty()  # Remove loading message after task completion

def cosine_similarity(vec_a, vec_b):
    """Compute cosine similarity between two vectors"""
    a, b = np.array(vec_a), np.array(vec_b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_articles(query, top_k=5, threshold=0.5):
    """Retrieve the most relevant articles for a query"""
    query_embedding = get_embedding(query)
    scored_articles = [
        (cosine_similarity(query_embedding, art["embedding"]), art)
        for art in article_corpus
    ]
    scored_articles.sort(reverse=True, key=lambda x: x[0])
    return [art for score, art in scored_articles if score >= threshold][:top_k]

def determine_question_type(query):
    """Decide if the query is Digital Twin-related"""
    keywords = ["digital twin", "digital twins", "digital shadow", "model", "system", "simulation", "real-time", "predictive", "modeling", "requirements"]
    if any(keyword in query.lower() for keyword in keywords):
        return "DT-related"
    return "general"

def gpt_answer_query(query):
    """Generate a GPT response with optional article references"""

    # Ensure the chat history is initialized
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Check if the last user message is the same as the new query (prevent duplicate entries)
    if not st.session_state["messages"] or st.session_state["messages"][-1]["role"] != "user":
        st.session_state["messages"].append({"role": "user", "content": [{"type": "text", "text": query}]})

    # Trim chat history (keep only last 10 exchanges)
    max_history_length = 10  
    chat_history = st.session_state["messages"][-max_history_length:]

    # Extract the last two user queries (if available)
    last_two_questions = [msg["content"][0]["text"] for msg in chat_history if msg["role"] == "user"][-2:]

    # Check if at least 1 out of the last 3 questions are DT-related
    dt_related_count = sum(1 for q in last_two_questions + [query] if determine_question_type(q) == "DT-related")

    is_dt_related = dt_related_count >= 1  # Consider it DT-related if at least 1 out of 3 questions are DT-related

    # Convert chat history into OpenAI format
    messages = [{"role": msg["role"], "content": content["text"]}
                for msg in chat_history for content in msg["content"]]

    relevant_articles = []

    if is_dt_related:
        relevant_articles = search_articles(query, top_k=5)

        if relevant_articles:
            references_text = "\n".join(f"- {art['title']}" for art in relevant_articles)
            system_prompt = f"""
            You are an AI expert in Digital Twins. Use the following research articles as reference but do not copy verbatim.

            **Relevant Articles:**
            {references_text}

            Answer based on the articles while considering previous conversation context.
            """

            # ✅ Make sure the system prompt is inserted correctly
            messages.insert(0, {"role": "system", "content": system_prompt})

    else:
        # If it's a general question, set a default assistant role
        messages.insert(0, {"role": "system", "content": "You are a helpful assistant."})

    # DEBUG: Check if articles were found**
    if is_dt_related:
        st.write(f"🔎 Found {len(relevant_articles)} relevant articles:")
        for art in relevant_articles:
            st.write(f"📄 {art['title']}")

    # Pass the full chat history to GPT
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.3,
    )

    answer = response.choices[0].message.content

    # Force Append References to Answer**
    if is_dt_related and relevant_articles:
        answer += f"\n\n**References Used:**\n{references_text}"

    # Prevent duplicate assistant messages
    if not st.session_state["messages"] or st.session_state["messages"][-1]["role"] != "assistant":
        st.session_state["messages"].append({"role": "assistant", "content": [{"type": "text", "text": answer}]})

    return answer

# Function to query and stream the response from the LLM (unchanged)
def stream_llm_response(client, model_params):
    response_message = ""
    for chunk in client.chat.completions.create(
        model=model_params["model"] if "model" in model_params else "gpt-4o-2024-05-13",
        messages=st.session_state.messages,
        temperature=model_params["temperature"] if "temperature" in model_params else 0.3,
        max_tokens=596,
        stream=True,
    ):
        response_message += chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""
        yield chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""
    st.session_state.messages.append({
        "role": "assistant", 
        "content": [{"type": "text", "text": response_message}]
    })

# Establish a Google Sheets connection
conn = GSheetsConnection(connection_name="gsheets")  # Match the name in secrets.toml

# Fetch existing data from all worksheets
profile_df = conn.read(worksheet="profile_data", usecols=list(range(9)), ttl=2000)  # Adjust `range` to match your columns
profile_df = profile_df.dropna(how="all")

scores_df = conn.read(worksheet="scores", usecols=list(range(6)), ttl=2000)  # Adjust `range` as per column count
scores_df = scores_df.dropna(how="all")

comments_df = conn.read(worksheet="comments", usecols=list(range(5)), ttl=2000)  # Adjust `range` as per column count
comments_df = comments_df.dropna(how="all")

# Generate unique session ID if not already present
if "unique_id" not in st.session_state:
    st.session_state["unique_id"] = str(uuid.uuid4())

# Sidebar navigation with logos
st.sidebar.markdown(
    """
    <style>
        .logo-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            width: 100%;
        }
        .logo-container img {
            max-width: 100%;
            margin-bottom: 15px; /* Space between images */
        }
        .dual-logo-container {
            display: flex;
            justify-content: space-between;
            width: 100%;
        }
        .dual-logo-container img {
            width: 48%; /* Make them fit within the sidebar */
        }
        .sidebar-text {
            margin-top: 20px; /* Space between logos and text */
        }
    </style>

    <div class="logo-container">
        <img src="https://cdn.brandfetch.io/idNkQH2wyM/w/401/h/84/theme/dark/logo.png?c=1bfwsmEH20zzEfSNTed">
        <div class="dual-logo-container">
            <img src="https://upload.wikimedia.org/wikipedia/fr/9/99/Logo_Polytechnique_Montr%C3%A9al.png">
            <img src="https://cdn.brandfetch.io/id79psAFnq/theme/dark/logo.svg?c=1bfwsmEH20zzEfSNTed">
        </div>
    </div>

    <div class="sidebar-text">
        <hr>
    </div>
    """,
    unsafe_allow_html=True
)

# System Classification function
def classify_system(scores):
    """
    Classifies a system based on rule-based logic using evaluation scores.
    """

    # Calculate average scores across all categories
    category_averages = {
        category: np.mean([np.mean(subcat) for subcat in subcategories.values()])
        for category, subcategories in scores.items()
    }

    # Calculate average scores for each subcategory
    subcategory_averages = {
        category: {
            subcat: np.mean(answers)
            for subcat, answers in subcategories.items()
        }
        for category, subcategories in scores.items()
    }

    # 1. Digital Twin
    if (
        category_averages.get("Caractéristiques principales du jumeau numérique", 0) > 3.5
        and subcategory_averages.get("Connectivité et Synchronisation", {}).get("Connexion Physique-Virtuelle", 0) > 3.5
        and subcategory_averages.get("Connectivité et Synchronisation", {}).get("Retour d'Information Virtuel-Physique", 0) >= 3
        and scores["Connectivité et Synchronisation"]["Synchronisation"][0] > 3
        and scores["Connectivité et Synchronisation"]["Synchronisation"][1] > 3
        and scores["Connectivité et Synchronisation"]["Synchronisation"][2] > 2
        and category_averages.get("Modélisation, Simulation et Aide à la Décision", 0) > 3.5
    ):
        classification = "Digital Twin"
        explanation = (
            "Votre système est qualifié de Jumeau Numérique car il répond aux caractéristiques "
            "obligatoires en termes de connectivité, de synchronisation et de support à la décision."
        )
        image_path = "images/digital_twin.png"

    # 2. Digital Shadow
    elif (
        category_averages.get("Caractéristiques principales du jumeau numérique", 0) > 3
        and subcategory_averages.get("Connectivité et Synchronisation", {}).get("Connexion Physique-Virtuelle", 0) > 3.5
        and subcategory_averages.get("Connectivité et Synchronisation", {}).get("Retour d'Information Virtuel-Physique", 0) < 2
        and scores["Connectivité et Synchronisation"]["Synchronisation"][0] > 3
        and scores["Connectivité et Synchronisation"]["Synchronisation"][2] > 2
        and category_averages.get("Modélisation, Simulation et Aide à la Décision", 0) > 3.5
    ):
        classification = "Digital Shadow"
        explanation = (
            "Votre système est un Ombre Numérique car il se concentre sur la collecte et la "
            "visualisation des données pour la simulation et la prise de décision, mais il manque "
            "de rétroaction et de contrôle en temps réel. Il peut être utile pour l'analyse et la "
            "prise de décision contextuelle, car il reste une représentation fidèle du jumeau physique."
        )
        image_path = "images/digital_shadow.png"

    # 3. Digital Model
    elif (
        category_averages.get("Caractéristiques principales du jumeau numérique", 0) > 3
        and subcategory_averages.get("Connectivité et Synchronisation", {}).get("Connexion Physique-Virtuelle", 0) < 2
        and subcategory_averages.get("Connectivité et Synchronisation", {}).get("Retour d'Information Virtuel-Physique", 0) < 2
        and subcategory_averages.get("Connectivité et Synchronisation", {}).get("Synchronisation", 0) < 2
        and category_averages.get("Modélisation, Simulation et Aide à la Décision", 0) > 3.5
    ):
        classification = "Digital model"
        explanation = (
            "Votre système est un Modèle Numérique car il manque à la fois de synchronisation en "
            "temps réel et de rétroaction. Il peut être utile pour la prise de décision contextuelle, "
            "mais nécessitera un entretien important puisqu'il n'évolue pas et n'interagit pas avec "
            "l'entité physique réelle qu'il représente."
        )
        image_path = "images/digital_model.png"

    # 4. Cyber-Physical System
    ### A virer potentiellement
    elif (
        category_averages.get("Caractéristiques principales du jumeau numérique", 0) < 3
        and subcategory_averages.get("Connectivité et Synchronisation", {}).get("Connexion Physique-Virtuelle", 0) > 3
        and subcategory_averages.get("Connectivité et Synchronisation", {}).get("Retour d'Information Virtuel-Physique", 0) > 3
        and subcategory_averages.get("Connectivité et Synchronisation", {}).get("Synchronisation", 0) > 3
        and category_averages.get("Modélisation, Simulation et Aide à la Décision", 0) < 2
    ):
        classification = "Cyber-Physical System"
        explanation = (
            "Votre système est classé comme un Système Cyber-Physique car il met l'accent sur "
            "l'intégration des composants physiques et numériques sans posséder pleinement les "
            "capacités d’un jumeau numérique. Il dispose d'une certaine représentation numérique, "
            "mais elle pourrait ne pas être complète. Concernant la connectivité et la synchronisation, "
            "des scores relativement élevés sont attendus, notamment si le système s'intègre bien avec "
            "des capteurs et des flux de données. En revanche, des scores plus faibles sont attendus "
            "en matière de modélisation, car le système ne propose pas de simulation complète ni de "
            "prise de décision proactive."
        )
        image_path = "images/cyber_physical.jpg"

    # 5. 3D Models & CAD
    elif (
        scores["Caractéristiques principales du jumeau numérique"]["Système Physique"][0] > 3
        and scores["Caractéristiques principales du jumeau numérique"]["Système Physique"][1] > 3
        and scores["Caractéristiques principales du jumeau numérique"]["Copie Virtuelle"][0] > 3
        and scores["Caractéristiques principales du jumeau numérique"]["Copie Virtuelle"][1] > 3
        and subcategory_averages.get("Connectivité et Synchronisation", {}).get("Connexion Physique-Virtuelle", 0) < 2
        and subcategory_averages.get("Connectivité et Synchronisation", {}).get("Retour d'Information Virtuel-Physique", 0) < 2
        and subcategory_averages.get("Connectivité et Synchronisation", {}).get("Synchronisation", 0) < 2
        and category_averages.get("Modélisation, Simulation et Aide à la Décision", 0) < 2
    ):
        classification = "3D Models & CAD"
        explanation = (
            "Votre système est principalement une représentation 3D ou un modèle CAO, axé sur "
            "la visualisation plutôt que sur l'intégration en temps réel."
        )
        image_path = "images/3d_model.webp"

    # 6. Digital Thread
    elif (
        category_averages.get("Gestion et Intégration des Données", 0) > 3
        and category_averages.get("Caractéristiques principales du jumeau numérique", 0) > 3
        and subcategory_averages.get("Connectivité et Synchronisation", {}).get("Connexion Physique-Virtuelle", 0) > 3
        and subcategory_averages.get("Connectivité et Synchronisation", {}).get("Retour d'Information Virtuel-Physique", 0) < 2
        and subcategory_averages.get("Connectivité et Synchronisation", {}).get("Synchronisation", 0) < 3
        and category_averages.get("Modélisation, Simulation et Aide à la Décision", 0) < 2
    ):
        classification = "Digital Thread"
        explanation = (
            "Votre système s'aligne avec le concept de Fil Numérique, intégrant des données sur "
            "le cycle de vie, mais manquant de simulation et de prise de décision autonome."
        )
        image_path = "images/digital_thread.jpeg"

    # 7. IoT or SCADA
    elif (
        scores["Caractéristiques principales du jumeau numérique"]["Système Physique"][0] >= 2
        and scores["Caractéristiques principales du jumeau numérique"]["Système Physique"][1] >= 2
        and scores["Caractéristiques principales du jumeau numérique"]["Copie Virtuelle"][0] > 3
        and scores["Caractéristiques principales du jumeau numérique"]["Copie Virtuelle"][1] > 3
        and subcategory_averages.get("Connectivité et Synchronisation", {}).get("Connexion Physique-Virtuelle", 0) > 3
        and subcategory_averages.get("Connectivité et Synchronisation", {}).get("Retour d'Information Virtuel-Physique", 0) < 2
        and subcategory_averages.get("Connectivité et Synchronisation", {}).get("Synchronisation", 0) > 2
        and category_averages.get("Modélisation, Simulation et Aide à la Décision", 0) < 2
    ):
        classification = "IoT or SCADA"
        explanation = (
            "Votre système correspond à la catégorie IoT ou SCADA en raison de sa forte dépendance "
            "à la collecte de données via des capteurs, sans intelligence complète de Jumeau Numérique. "
            "Il ne possède pas de modules de simulation, de moteurs de calcul ou d’autonomie pour "
            "interagir avec l'entité physique."
        )
        image_path = "images/iot_scada.png"

    # 8. Other / Not a Digital Twin
    else:
        classification = "Other / Not a Digital Twin"
        explanation = (
            "Votre système ne répond pas aux caractéristiques fondamentales d'un Jumeau Numérique, "
            "mais peut appartenir à une autre catégorie de technologies numériques."
        )
        image_path = "images/other.png"

    return classification, explanation, image_path

# Define the evaluation framework with fuzzy categories

## Ajouter une question par rapport au niveau de synchronisation de l'outil vis à vis à l'application du jumeau numérique.
## Defenitly need to revisit this article for further questions : Digital Twins: A Maturity Model for Their Classification and Evaluation

evaluation_framework = {
    "Caractéristiques principales du jumeau numérique": {
        "description": "Les Jumeaux Numériques sont des représentations virtuelles d'objets, systèmes ou processus physiques, permettant un échange bidirectionnel de données pour la supérvision en temps réel, la simulation et la prise de décision. Les composants clés d'un Jumeau Numérique sont le système physique, sa copie virtuelle et le transfert de données qui les relie.",
        "subcategories": [
            {
                "subcategory": "Système Physique",
                "questions": [
                    {"question": "Dans quelle mesure le périmètre du système physique (ex. : entrepôt, supply chain) est-il bien défini ? ", "type": "fuzzy"},
                    {"question": "Dans quelle mesure les différentes composantes du système (ex. : équipements, processus) sont-elles clairement identifiées et organisées selon leur rôle dans le WMS ?", "type": "fuzzy"},
                    {"question": "Dans quelle mesure les conditions physiques (ex. : température, pression, environnement opérationnel) qui influencent l’entrepôt sont-elles bien accessible sur le WMS ?", "type": "fuzzy"}
                ]
            },
            {
                "subcategory": "Copie Virtuelle",
                "questions": [
                    {"question": "Le WMS représente-t-il fidèlement l’entrepôt et ses opérations ?", "type": "fuzzy"},
                    {"question": "Le niveau de détail est-il suffisant pour suivre et comprendre les flux logistiques et les interactions entre les composants de l'entrepôt ?", "type": "fuzzy"},
                    {"question": "Le WMS comprend-il une interface utilisateur intuitive pour permettre le pilotage de l'activité, l'accès aux données, l’analyse et l’interaction ou l’exécution d’expériences ?", "type": "fuzzy"}
                ]
            }
        ]
    },
    "Connectivité et Synchronisation": {
        "description": "Une caractéristique fondamentale des Jumeaux Numériques est leur capacité à maintenir des connexions dynamiques et bidirectionnelles entre les entités physiques et virtuelles. Cela implique d'assurer la synchronisation via des flux de données en temps réel ou quasi réel pour soutenir les objectifs opérationnels et stratégiques, ainsi que la capacité du système à réagir et interagir avec l'entité physique lorsque nécessaire (dans son périmètre d'application).",
        "subcategories": [
            {
                "subcategory": "Connexion Physique-Virtuelle",
                "questions": [
                    {"question": "Dans quelle mesure la mise à jour des données d'activité de l'entrepôt vers le WMS sont-elles automatisées ?", "type": "fuzzy"},
                    {"question": "À quelle fréquence les données d'activité de l'entrepôt sont-elles envoyées au WMS ? (1 = jamais, 5 = en temps réel)", "type": "fuzzy"},
                    {"question": "Dans quelle mesure Le WMS est-il (peut il être) bien connecté aux autres outils numériques de l'entrepôt (ex. : SAP, capteurs IoT, cloud) ? (1 = aucune interopérabilité, 5 = communication avec d'autres outils)", "type": "fuzzy"}
                ]
            },
            {
                "subcategory": "Retour d'Information Virtuel-Physique",
                "questions": [
                    {"question": "Un WMS peut-il prendre des décisions en temps réel pour optimiser les opérations de l’entrepôt ? (1 = aucune prise de décision, 3 = support de prise de décision, 5 = analyse et prise de décisions autonomes)", "type": "fuzzy"},
                    {"question": "Dans quelle mesure un WMS peut-il déclencher automatiquement des actions dans l’entrepôt (ex. : Lancement de préparation, ajustement des stocks, guidage les opérateurs, alert en cas d’anomalie) ? (1 = aucune réaction possible, 5 = envoi de commandes de contrôle ou notifications aux opérateurs)", "type": "fuzzy"}
                ]
            },
            {
                "subcategory": "Synchronisation",
                "questions": [
                    {"question": "La méthode de connexion entre l’entrepôt physique et le WMS est-elle bien définie (ex. : PDA, capteurs, infrastructure IT et matériel) (1 = pas de transmission de données, 5 = transmission automatique de données l'entrepôt vers le WMS)?", "type": "fuzzy"},
                    {"question": "Le délai de mise à jour des données est-il adapté aux besoins opérationnels du WMS et aux exigences de la prise de décision en entrepôt?", "type": "fuzzy"},
                    {"question": "Le WMS permet-il d’analyser l’historique, l’état actuel et les prédictions des opérations (1 = analyse de l'historique, 5 = analyse prédictive de l'entrepôt)?", "type": "fuzzy"}
                ]
            }
        ]
    },
    "Modélisation, Simulation et Aide à la Décision": {
        "description": "Les Jumeaux Numériques permettent des capacités prédictives et prescriptives grâce à des moteurs de calcul, fournissant des analyses exploitables et des stratégies d'optimisation. Ces capacités facilitent la transition de la prise de décision réactive à proactive.",
        "subcategories": [
            {
                "subcategory": "Modélisation et Scénarios Prospectifs",
                "questions": [
                    {"question": "Le WMS dispose-t-il d'un moteur de calcul ou de simulation pour tester différents scénarios et optimiser les décisions ? (1 = Pas de calcul local, 5 = Calcul possible)", "type": "fuzzy"},
                    {"question": "Le WMS peut-il simuler des situations hypothétiques (ex. : pic d’activité, perturbations, changements de stock) ?", "type": "fuzzy"}
                ]
            },
            {
                "subcategory": "Optimisation et Prise de Décision",
                "questions": [
                    {"question": "Le WMS peut-il appliquer des algorithmes pour améliorer la gestion des stocks, les flux logistiques ou l’efficacité énergétique ?", "type": "fuzzy"},
                    {"question": "Le WMS fournit-il des recommandations claires aux opérateurs ou gestionnaires d’entrepôt ?", "type": "fuzzy"}
                ]
            }
        ]
    },
    "Gestion et Intégration des Données": {
        "description": "Les Jumeaux Numériques reposent sur des infrastructures robustes de collecte, d'intégration et de traitement des données pour assurer des opérations en temps réel fluides. Cela inclut les dispositifs IoT, l'informatique en cloud/edge et la compatibilité avec les systèmes d'entreprise.",
        "subcategories": [
            {
                "subcategory": "Intégration Systémique",
                "questions": [
                    {"question": "Le WMS gère-t-il efficacement différents types de données (temps réel vs historiques, structurées vs non structurées) ?", "type": "fuzzy"},
                    {"question": "Le WMS peut-il évoluer et gérer des volumes de données croissants ?", "type": "fuzzy"}
                ]
            },
            {
                "subcategory": "Collecte et Traitement des Données",
                "questions": [
                    {"question": "Dans quelle mesure le WMS gère-t-il bien différents formats et sources de données (ex. : fichiers Excel, images, bases de données, capteurs) ?", "type": "fuzzy"},
                    {"question": "Le WMS utilise-t-il plusieurs sources d’information pour une meilleure analyse des opérations ?", "type": "fuzzy"}
                ]
            }
        ]
    },
    "Apprentissage, Adaptabilité et Autonomie": {
        "description": "Un Jumeau Numérique mature exploite l'intelligence artificielle et l'apprentissage automatique pour s'améliorer, reconnaître les changements de contexte et adapter ses modèles de manière autonome. Cette adaptabilité garantit la scalabilité et la pertinence du système tout au long de son cycle de vie.",
        "subcategories": [
            {
                "subcategory": "Connaissance du Contexte",
                "questions": [
                    {"question": "Le WMS détecte-t-il automatiquement les changements dans l’environnement de l’entrepôt ?", "type": "fuzzy"},
                    {"question": "Dans quelle mesure le WMS prend-il en compte les interactions entre équipements, les événements imprévus et les incertitudes ?", "type": "fuzzy"}
                ]
            },
            {
                "subcategory": "Capacités d'Apprentissage",
                "questions": [
                    {"question": "Le WMS est-il capable d’apprendre de ses propres expériences, des données et de s’améliorer avec le temps ? (1 = Aucune intelligence, 5 = Apprentissage entièrement autonome)", "type": "fuzzy"},
                    {"question": "Dans quelle mesure Le système utilise-t-il l’IA ou l’apprentissage automatique pour optimiser la gestion de l’entrepôt ?", "type": "fuzzy"},
                    {"question": "Dans quelle mesure Les décisions du WMS sont-elles claires et compréhensibles pour les opérateurs ? (1 = Pas de décision, 5 = Décisions avec raisonnement logique)", "type": "fuzzy"}
                ]
            },
            {
                "subcategory": "Adaptabilité et Évolution",
                "questions": [
                    {"question": "Le WMS peut-il facilement intégrer de nouveaux équipements, technologies ou processus logistiques ?", "type": "fuzzy"},
                    {"question": "Le WMS peut il être utilisé tout au long du cycle de vie de l’entrepôt ?", "type": "fuzzy"}
                ]
            },
            {
                "subcategory": "Autonomie",
                "questions": [
                    {"question": "Dans quelle mesure le WMS est-il capable d’analyser des situations ? (1 = Aucune analyse, 5 = Analyse de cas concret pour prise de décision) ", "type": "fuzzy"},
                    {"question": "Le WMS peut-il prendre et exécuter des décisions de manière autonome dans son périmètre d'application défini ? (1 = Lancement manuel à travers le WMS, 5 = Execution automatique après paramétrage)", "type": "fuzzy"}
                ]
            }
        ]
    },
    "Fidélité et Validation": {
        "description": "Les Jumeaux Numériques visent des représentations haute-fidélité tout en maintenant une efficacité computationnelle optimale. La validation garantit leur fiabilité et leur alignement avec les comportements physiques, ce qui est essentiel pour la confiance des parties prenantes.",
        "subcategories": [
            {
                "subcategory": "Niveau d'Abstraction",
                "questions": [
                    {"question": "Les calculs (ou simulations) du WMS correspondent-elles au comportement réel de l’entrepôt ? (face aux mêmes stimuli) ?", "type": "fuzzy"},
                    {"question": "Le WMS donne-t-il des résultats reproductibles avec les mêmes données d’entrée ?", "type": "fuzzy"}
                ]
            },
            {
                "subcategory": "Vérification et Retour d'Information",
                "questions": [
                    {"question": "Le WMS est-il régulièrement testé et comparé aux performances réelles ?(ex. : tests, analyses de sensibilité, comparaisons avec le monde réel)", "type": "fuzzy"},
                    {"question": "Les données de l'entrepôt sont-elles utilisées pour affiner et améliorer le WMS ?", "type": "fuzzy"}
                ]
            }
        ]
    },
    "Services du Jumeau Numérique": {
        "description": "L'utilité fonctionnelle des Jumeaux Numériques se mesure à travers leurs capacités de service, telles que la surveillance en temps réel, la maintenance prédictive et l'optimisation opérationnelle, visant à améliorer la performance et la résilience du système.",
        "subcategories": [
            {
                "subcategory": "Surveillance en Temps Réel",
                "questions": [
                    {"question": "Le WMS permet-il un suivi en temps réel des indicateurs clés (ex. : performance, consommation d’énergie, erreurs) ?", "type": "fuzzy"},
                    {"question": "Le WMS est-il accessible et utilisable sur différents appareils et plateformes (ex. : PC, mobile, tablette) ?", "type": "fuzzy"}
                ]
            },
            {
                "subcategory": "Optimisation et Prédiction",
                "questions": [
                    {"question": "Le WMS peut-il anticiper des tendances ou événements impactant l’entrepôt (ex. : pics d’activité, pannes) ?", "type": "fuzzy"},
                    {"question": "Dans quelle mesure le WMS intègre-t-il des outils d’analyse prédictive pour améliorer la gestion de l’entrepôt ?", "type": "fuzzy"},
                    {"question": "Dans quelle mesure le WMS propose-t-il des recommandations automatisées pour optimiser les opérations ? ", "type": "fuzzy"}
                ]
            }
        ]
    },
    "Maturité Technologique": {
        "description": "Le déploiement des Jumeaux Numériques repose sur l'intégration de technologies avancées telles que l'IoT, l'informatique en cloud, l'IA/l'apprentissage automatique et la cybersécurité. La scalabilité et la conformité aux normes de protection des données sont également des considérations essentielles.",
        "subcategories": [
            {
                "subcategory": "Technologies Facilitatrices",
                "questions": [
                    {"question": "Le WMS intègre-t-il des technologies avancées comme l’IoT, le cloud computing ou l’IA ? ", "type": "fuzzy"},
                    {"question": "Le WMS est-il accessible aux experts métier sans nécessiter une expertise technique approfondie en programmation ? ", "type": "fuzzy"}
                ]
            },
            {
                "subcategory": "Sécurité et Confidentialité",
                "questions": [
                    {"question": "Dans quelle mesure le WMS empêche-t-il les accès non autorisés ?", "type": "fuzzy"},
                    {"question": "Le WMS applique-t-il des mesures robustes pour garantir la confidentialité des données ?", "type": "fuzzy"}
                ]
            }
        ]
    }
}

# Track submission state to avoid multiple submissions
if "has_submitted" not in st.session_state:
    st.session_state["has_submitted"] = False

# Submission Function
def submit_evaluation():
    """Handles submission logic to store data and prevent multiple submissions."""
    if st.session_state["has_submitted"]:
        st.warning("Vous avez déjà soumis vos réponses. Merci !")
        return

    timestamp = pd.Timestamp.now().isoformat()

    # Prepare profile_data with unique_id
    profile_data = {**st.session_state["profile_data"], "timestamp": timestamp, "unique_id": st.session_state["unique_id"]}

    # Prepare scores
    scores_data = []
    for category, subcategories in st.session_state["scores"].items():
        for subcat, answers in subcategories.items():
            for idx, score in enumerate(answers):
                subcategory_data = next(
                    (sub for sub in evaluation_framework[category]["subcategories"] if sub["subcategory"] == subcat),
                    None
                )
                question_text = subcategory_data["questions"][idx]["question"] if subcategory_data else "Unknown Question"

                scores_data.append({
                    "timestamp": timestamp,
                    "unique_id": st.session_state["unique_id"],
                    "category": category,
                    "subcategory": subcat,
                    "question": question_text,
                    "score": score
                })

    # Prepare comments
    comments_data = []
    for category, subcategories in st.session_state["comments"].items():
        for subcat, comment in subcategories.items():
            comments_data.append({
                "timestamp": timestamp,
                "unique_id": st.session_state["unique_id"],
                "category": category,
                "subcategory": subcat,
                "comment": comment
            })

    # Convert to DataFrames
    profile_data_df = pd.DataFrame([profile_data])
    scores_data_df = pd.DataFrame(scores_data)
    comments_data_df = pd.DataFrame(comments_data)

    # Merge with existing data
    updated_profiles = pd.concat([profile_df, profile_data_df], ignore_index=True)
    updated_scores = pd.concat([scores_df, scores_data_df], ignore_index=True)
    updated_comments = pd.concat([comments_df, comments_data_df], ignore_index=True)

    # Update Google Sheets
    conn.update(worksheet="profile_data", data=updated_profiles)
    conn.update(worksheet="scores", data=updated_scores)
    conn.update(worksheet="comments", data=updated_comments)

    st.success("✅ Votre retour a été soumis avec succès !")
    st.session_state["has_submitted"] = True  # Prevent further submissions

# Ensure navigation state exists in session
if "navigation" not in st.session_state:
    st.session_state["navigation"] = "Profile Identification"  # Default page

# Initialize navigation state
if "redirect_to" not in st.session_state:
    st.session_state["redirect_to"] = None

# Ensure session state for navigation
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Profile Identification"  # Default start page

# List of pages
pages = ["Profile Identification"] + list(evaluation_framework.keys()) + ["Summary", "Chatbot"]

# Ensure current_page exists in the list before calling .index()
if st.session_state["current_page"] not in pages:
    st.session_state["current_page"] = "Profile Identification"  # Default to the first page

# Sidebar navigation
sidebar_page = st.sidebar.radio(
    "Select a Category",
    options=pages,
    index=pages.index(st.session_state["current_page"])  # Now it's correctly formatted
)

# Sync sidebar selection with button navigation
st.session_state["current_page"] = sidebar_page

# Set the active page
page = st.session_state["current_page"]

# Check if the page has changed and trigger scroll
if "last_page" not in st.session_state:
    st.session_state["last_page"] = page  # Initialize with first page

if st.session_state["last_page"] != page:
    if page == "Chatbot":
        st.session_state.scroll_to_bottom = True  # Scroll to bottom for chatbot
    else:
        st.session_state.scroll_to_top = True  # Scroll to top for all other pages

st.session_state["last_page"] = page  # Update last visited page

if st.session_state.redirect_to:
    page = st.session_state.redirect_to
    st.session_state.redirect_to = None

def profile_identification():
    # Ensure profile_data exists in session state
    if "profile_data" not in st.session_state:
        st.session_state["profile_data"] = {
            "field_of_work": "Research",
            "years_experience": 0,
            "current_system": "",
            "position": "",
            "country": "",
            "department": "",
            "comments": ""
        }
    
    # Remplir les champs avec les valeurs existantes de l'état de session
    st.write("Avant de commencer, apprenons à mieux nous connaître 🙂 Les résultats de ce questionnaire seront collectés de manière anonyme à des fins de recherche. Cela vous prendra environ 15 minutes à compléter. Les résultats du questionnaire vous permettront également d'échanger avec un GPT spécialisé en littérature sur les jumeaux numériques. N'hésitez donc pas à ajouter autant de commentaires que nécessaire pour obtenir des réponses concrètes.")
    
    st.write("""
    Toutes les questions de ce questionnaire doivent être évaluées sur une échelle de 1 à 5, reflétant le degré d'adéquation du système d'information ou du WMS aux critères proposés :
    - **1** : Le système ne répond pas du tout à cette exigence.
    - **2** : Le système y répond partiellement, mais de manière très limitée ou inefficace.
    - **3** : Le critère est pris en charge, mais avec des lacunes ou des limitations significatives.
    - **4** : L'exigence est bien remplie et le système est fonctionnel pour un usage quotidien.
    - **5** : Le critère est pleinement intégré, démontrant une prise en charge avancée et efficace.

    Veuillez évaluer chaque question de manière objective afin d'obtenir une analyse pertinente de votre système.
    """)

    st.session_state["profile_data"]["field_of_work"] = st.radio(
        "Quel est votre domaine d'activité ?", 
        ["Recherche", "Industrie", "logistique et supply chain"], 
        index=["Recherche", "Industrie", "logistique et supply chain"].index(st.session_state["profile_data"].get("field_of_work", "Recherche"))
    )
    st.session_state["profile_data"]["years_experience"] = st.slider(
        "Depuis combien d'années travaillez-vous sur les Systèmes d'Information (SI) ou les Systèmes de Gestion d'Entrepôt (WMS) ?", 
        0, 50, step=1, 
        value=st.session_state["profile_data"].get("years_experience", 0)
    )
    st.session_state["profile_data"]["current_system"] = st.text_input(
        "Quel est le nom du SI ou du WMS que vous utilisez (si applicable) ?", 
        value=st.session_state["profile_data"].get("current_system", "")
    )
    st.session_state["profile_data"]["position"] = st.text_input(
        "Quel est votre poste actuel ?", 
        value=st.session_state["profile_data"].get("position", "")
    )
    st.session_state["profile_data"]["country"] = st.text_input(
        "Dans quel pays êtes-vous basé(e) ?", 
        value=st.session_state["profile_data"].get("country", "")
    )
    st.session_state["profile_data"]["department"] = st.text_input(
        "À quel département êtes-vous rattaché(e) (ex. : R&D, Logistique) ?", 
        value=st.session_state["profile_data"].get("department", "")
    )
    st.session_state["profile_data"]["comments"] = st.text_area(
        "Avez-vous des commentaires ou des remarques supplémentaires à partager ?", 
        value=st.session_state["profile_data"].get("comments", "")
    )
    return st.session_state["profile_data"]

# Radar Chart Function using Plotly
def plot_radar_chart(data, categories):
    fig = go.Figure()

    # Add radar trace
    fig.add_trace(go.Scatterpolar(
        r=data, theta=categories, fill='toself', name='Average Scores'))

    # Update layout for aesthetics
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5],  # Adjust the range as per your score scale
            ),
        ),
        showlegend=False,
        title="Radar Chart of Average Scores",
    )

    # Display the radar chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

# Initialize session state
if "scores" not in st.session_state:
    st.session_state.scores = {
        category: {item["subcategory"]: [None] * len(item["questions"]) for item in data["subcategories"]}
        for category, data in evaluation_framework.items()
    }
if "comments" not in st.session_state:
    st.session_state.comments = {
        category: {item["subcategory"]: "" for item in data["subcategories"]}
        for category, data in evaluation_framework.items()
    }
if "profile_data" not in st.session_state:
    st.session_state.profile_data = {}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Fonction de validation
def validate_all_answers():
    """Vérifie que toutes les questions sont complétées en recherchant des valeurs None."""
    for category, subcategories in st.session_state.scores.items():
        for subcat, answers in subcategories.items():
            if None in answers:  # Vérifier les valeurs non initialisées (None)
                return False, f"Veuillez compléter toutes les questions dans '{category}' - '{subcat}'."
    return True, ""

# Update session state when navigating via radio
st.session_state["current_page"] = page

# Track last visited page
if "last_page" not in st.session_state:
    st.session_state["last_page"] = page

# Scroll to top when switching sections (except Chatbot)
if st.session_state["last_page"] != page:
    st.session_state["last_page"] = page
    if page != "Chatbot":
        st.markdown(
            """
            <script>
            window.scrollTo(0, 0);
            </script>
            """,
            unsafe_allow_html=True
        )

# Scroll to bottom when inside Chatbot
if page == "Chatbot":
    st.markdown(
        """
        <script>
        window.scrollTo(0, document.body.scrollHeight);
        </script>
        """,
        unsafe_allow_html=True
    )

# Affichage du contenu principal
st.title("Cadre d'Évaluation des Jumeaux Numériques")

if page == "Profile Identification":
    profile_data = profile_identification()
    st.session_state["profile_data"] = profile_data

elif page in evaluation_framework.keys():

    if st.session_state.scroll_to_top:
        scroll_to_here(0, key="top")  # Scroll to top
        st.session_state.scroll_to_top = False  # Reset state
    
    # Render evaluation questions
    st.subheader(f"Catégorie : {page}")
    st.write(evaluation_framework[page]["description"])

    for subcategory in evaluation_framework[page]["subcategories"]:
        st.write(f"### {subcategory['subcategory']}")
        for i, question in enumerate(subcategory["questions"]):
            if st.session_state.scores[page][subcategory["subcategory"]][i] is None:
                st.session_state.scores[page][subcategory["subcategory"]][i] = 0
        
            # Replace slider with a horizontal radio button
            st.session_state.scores[page][subcategory["subcategory"]][i] = st.radio(
                question["question"], 
                options=[1, 2, 3, 4, 5], 
                horizontal=True,  # Display options in a row
                key=f"{page}_{subcategory['subcategory']}_{i}",
                index=st.session_state.scores[page][subcategory["subcategory"]][i] - 1 if st.session_state.scores[page][subcategory["subcategory"]][i] else 0
            )

        comment = st.text_area(
            f"Commentaires pour {subcategory['subcategory']}",
            key=f"comment_{page}_{subcategory['subcategory']}",
            value=st.session_state.comments[page][subcategory["subcategory"]]
            )
        st.session_state.comments[page][subcategory["subcategory"]] = comment
    
    if(page == "Maturité Technologique"):
        if st.button("✅ Soumettre maintenant"):
            submit_evaluation()
            # Create summary DataFrame with comments
            summary_data = []
            for category, subcategories in st.session_state.scores.items():
                for subcat, answers in subcategories.items():
                    avg_score = np.mean([a for a in answers if a is not None]) if answers else 0
                    summary_data.append({
                        "Category": category,
                        "Subcategory": subcat,
                        "Average Score": avg_score,
                        "Comments": st.session_state.comments[category][subcat]  # Include comments
                    })
            # Convert summary data to a DataFrame and display as a table
            summary_df = pd.DataFrame(summary_data)
            st.session_state["summary_df"] = summary_df


elif page == "Summary":
    st.subheader("Résumé de l'Évaluation")

    # Initialize summary data list
    summary_data = []

    # Create summary DataFrame with comments
    summary_data = []
    for category, subcategories in st.session_state.scores.items():
        for subcat, answers in subcategories.items():
            avg_score = np.mean([a for a in answers if a is not None]) if answers else 0
            summary_data.append({
                "Category": category,
                "Subcategory": subcat,
                "Average Score": avg_score,
                "Comments": st.session_state.comments[category][subcat]  # Include comments
            })

    # Convert summary data to a DataFrame and display as a table
    summary_df = pd.DataFrame(summary_data)
    st.table(summary_df[["Category", "Subcategory", "Average Score"]])

    # Check if all questions have been answered
    valid, error_message = validate_all_answers()

    if not valid:
        st.error(error_message)
    else:
        # Submission button at the beginning of the Summary Page
        if not st.session_state["has_submitted"]:  
            if st.button("✅ Soumettre vos réponses"):
                submit_evaluation()
                st.session_state["summary_df"] = summary_df
        
        st.write("""
            ### Prêt à continuer ?
            Le tableau ci-dessus résume les scores d'évaluation pour chaque catégorie et sous-catégorie.
            Le graphique radar ci-dessous visualise les scores moyens par catégorie.
            Cliquez sur le bouton 'Soumettre et Continuer' pour partager vos résultats à des fins de recherche. 😊
        """)

        # Prepare data for radar chart
        radar_data = summary_df.groupby("Category")["Average Score"].mean().tolist()
        radar_categories = summary_df["Category"].unique()

        # Plot radar chart
        plot_radar_chart(radar_data, radar_categories)

        # System classification
        st.markdown("---")
        st.subheader("Classification du Système")

        classification, explanation, image_path = classify_system(st.session_state.scores)

        st.subheader(f"Classification du Système : {classification}")

        # Display Image
        if os.path.exists(image_path):
            st.image(image_path, caption=classification,  use_container_width=True)
        else:
            st.warning(f"Image not found: {image_path}")

        # Display Explanation
        st.write(f"**Explication :** {explanation}")

        # if the ssytem really is a Digital twin (in which case ... props to you!)
        if(classification == "Digital Twin"):
            # Display further explanation
            st.write(f"Il n'existe pas de définition universelle des Jumeaux Numériques, ce qui souligne encore plus le besoin d'un cadre standardisé. Les caractéristiques fondamentales de cette technologie sont bien définies. Toutefois, différents niveaux de maturité peuvent encore être identifiés dans le paradigme du Jumeau Numérique. Voici une analyse plus approfondie de la maturité de votre Jumeau Numérique :")

        # Final Submission Button at the End of Summary Page
        if not st.session_state["has_submitted"]:
            if st.button("✅ Soumettre et passer au Chatbot"):
                submit_evaluation()
        
        if st.session_state["has_submitted"]:    
            st.success("🎉 Vous pouvez maintenant discuter avec notre chatbot !")
            st.session_state["summary_df"] = summary_df

            # Auto-generate first chatbot question
            summary_text = ""
            for _, row in summary_df.iterrows():
                summary_text += f"- **Category**: {row['Category']}\n"
                summary_text += f"  - **Subcategory**: {row['Subcategory']}\n"
                summary_text += f"  - **Score**: {row['Average Score']}\n"
                if row["Comments"]:
                    summary_text += f"  - **Comments**: {row['Comments']}\n"
                summary_text += "\n"  # Blank line between entries

            initial_question = f"""
            Based on the Digital Twin Evaluation results:
            {summary_text}
    
            Please analyze and provide insights on how well this system aligns with Digital Twin principles.
            Highlight strengths, weaknesses, and areas for improvement.
            """
            st.session_state["initial_chatbot_question"] = initial_question

# Chatbot Page
elif page == "Chatbot":
    st.header("Digital Twin Chatbot 🤖")

    # Mark this section with an ID for scrolling
    st.markdown('<div id="chatbot-section"></div>', unsafe_allow_html=True)

    # --- Side Bar ---
    with st.sidebar:
        default_openai_api_key =  st.secrets["openai"]["openai_api_key"] if st.secrets["openai"]["openai_api_key"] is not None else ""  # only for development environment, otherwise it should return None
        with st.popover("OpenAI API Key"):
            openai_api_key = st.text_input("Introduce your OpenAI API Key (https://platform.openai.com/)", value=default_openai_api_key, type="password")

        if not (openai_api_key == "" or openai_api_key is None or "sk-" not in openai_api_key):
            st.divider()

    # Checking if the user has introduced the OpenAI API Key, if not, a warning is displayed
    if openai_api_key == "" or openai_api_key is None or "sk-" not in openai_api_key:
        st.write("#")
        st.warning("⬅️ Please introduce your OpenAI API Key (make sure to have funds) to continue...")
    else:
        client = OpenAI(api_key=openai_api_key)    
        
        # Ensure messages state exists
        if "messages" not in st.session_state:
            st.session_state["messages"] = []
    
        # --- Process Initial Chatbot Question ---
        if "initial_chatbot_question" in st.session_state and not st.session_state["messages"]:
            initial_question = st.session_state.pop("initial_chatbot_question")
            initial_response = gpt_answer_query(initial_question)

            st.session_state["messages"] = [
                {"role": "user", "content": [{"type": "text", "text": initial_question}]},
                {"role": "assistant", "content": [{"type": "text", "text": initial_response}]},
            ]

        # Ensure the summary data exists
        if "summary_df" not in st.session_state or st.session_state["summary_df"].empty:
            st.error("Summary data is missing. Please complete the evaluation first and submit your answers.")
            st.stop()

        # Define model parameters before usage
        model_params = {
            "model": "gpt-4o-mini",
            "temperature": 0.2,
        }

        # Display previous messages in the chat
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                for content in message["content"]:
                    if content["type"] == "text":
                        st.write(content["text"])

        # Side bar model options and inputs
        with st.sidebar:

            model = st.selectbox("Select a model:", [
                "gpt-3.5-turbo",
                "gpt-4o-2024-08-06", 
                "gpt-4o-mini-realtime-preview", 
                "dall-e-2", 
                "davinci-002",
            ], index=0)

            with st.popover("Model parameters"):
                model_temp = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.3, step=0.1)

            model_params = {
                "model": model,
                "temperature": model_temp,
            }

            def reset_conversation():
                if "messages" in st.session_state and len(st.session_state.messages) > 0:
                    st.session_state.pop("messages", None)

            st.button(
                "🗑️ Reset conversation", 
                on_click=reset_conversation,
            )

            st.divider()

        # Only process user input if it's non-empty
        prompt = st.chat_input("Hi! Ask me anything...")

        if prompt and prompt.strip():
            # Ensure messages list exists
            if "messages" not in st.session_state:
                st.session_state["messages"] = []

            # Check if the last message is already a user message (prevents duplication)
            if not st.session_state.messages or st.session_state.messages[-1]["role"] != "user":
                st.session_state.messages.append({
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
                })

                # Display the user's message immediately
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Make sure the assistant's response is displayed **right after the query**
                with st.spinner("Thinking..."):
                    answer = gpt_answer_query(prompt)  # Compute the response

                # Add the assistant's response to session state **before** rerun
                if not st.session_state.messages or st.session_state.messages[-1]["role"] != "assistant":
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": answer}]
                    })

                # Immediately display the assistant's response**
                with st.chat_message("assistant"):
                    st.markdown(answer)

                # Force a UI rerun so that Streamlit updates correctly**
                st.rerun()
        else:
            # Optionally show a placeholder or do nothing
            st.write("Please type something to get an answer.")

# --- Move Navigation Buttons to Bottom and Center Them ---
st.markdown("---")  # Adds a horizontal separator

# Create three columns and place buttons in the middle one
col1, col2, col3 = st.columns([1, 2, 1])

with col2:  # Center column
    button_container = st.container()
    with button_container:
        col_left, col_center, col_right = st.columns([1, 2, 1])

        with col_left:
            if pages.index(page) > 0:  # If not the first page
                if st.button("⬅️"):
                    st.session_state["current_page"] = pages[pages.index(page) - 1]
                    st.rerun()

        with col_right:
            if pages.index(page) < len(pages) - 1:  # If not the last page
                if st.button("➡️"):
                    st.session_state["current_page"] = pages[pages.index(page) + 1]
                    st.rerun()

# Footer section (Always displayed at the bottom of every page)
# Custom CSS for fixed right-aligned footer with padding
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: white;
        text-align: right;
        padding: 10px 20px;
        font-size: 12px;
        box-shadow: 0px -2px 5px rgba(0, 0, 0, 0.1);
    }
    .footer-center { text-align: center; }
    </style>
    
    <div class="footer">
        <strong>Notice:</strong> This app is developed in collaboration between 
        Square Management and École Nationale Supérieure d'Arts et Métiers
        for the <strong>Digital Twin Project</strong>.<br>
        <div class="footer-center">📩 Contact us: adnane.drissi_elbouzidi@ensam.eu <br> Adnane Drissi Elbouzidi</div>
    </div>
    """,
    unsafe_allow_html=True
)
