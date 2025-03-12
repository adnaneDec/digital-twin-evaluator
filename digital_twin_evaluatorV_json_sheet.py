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
    import numpy as np

    # Calculate average scores across all categories
    category_averages = {category: np.mean([np.mean(subcat) for subcat in subcategories.values()]) 
                         for category, subcategories in scores.items()}

    # Calculate average scores for each subcategory
    subcategory_averages = {
        category: {
            subcat: np.mean(answers)
            for subcat, answers in subcategories.items()
        }
        for category, subcategories in scores.items()
    }

    # Classification Logic (Modify these thresholds based on your framework)
    if category_averages.get("Core Digital Twin Characteristics", 0) > 3.5 and \
        subcategory_averages.get("Connectivity and Synchronization", {}).get("Physical-to-Virtual Connection", 0)> 3.5 and \
        subcategory_averages.get("Connectivity and Synchronization", {}).get("Virtual-to-Physical Feedback", 0)>= 3 and \
        scores["Connectivity and Synchronization"]["Synchronization"][0] > 3 and \
        scores["Connectivity and Synchronization"]["Synchronization"][1] > 3 and \
        scores["Connectivity and Synchronization"]["Synchronization"][2] > 2 and \
       category_averages.get("Modeling, Simulation and Decision Support", 0) > 3.5:
        classification = "Digital Twin"
        explanation = "Your system qualifies as a Digital Twin because it meets mandatory characteristics in terms of connectivity, synchronization, and decision support."
        image_path = "images/digital_twin.png"

    elif category_averages.get("Core Digital Twin Characteristics", 0) > 3 and \
         subcategory_averages.get("Connectivity and Synchronization", {}).get("Physical-to-Virtual Connection", 0)> 3.5 and \
         subcategory_averages.get("Connectivity and Synchronization", {}).get("Virtual-to-Physical Feedback", 0) < 2 and \
         scores["Connectivity and Synchronization"]["Synchronization"][0] > 3 and \
         scores["Connectivity and Synchronization"]["Synchronization"][2] > 2 and \
        category_averages.get("Modeling, Simulation and Decision Support", 0) > 3.5:
        classification = "Digital Shadow"
        explanation = "Your system is a Digital Shadow because it focuses on data collection and visualization for simulation and decision making but lacks real-time feedback and control. it can be useful for analysis and situational decision making as it remains an accurate representation of the physical twin."
        image_path = "images/digital_shadow.png"

    elif category_averages.get("Core Digital Twin Characteristics", 0) > 3 and \
         subcategory_averages.get("Connectivity and Synchronization", {}).get("Physical-to-Virtual Connection", 0)< 2 and \
         subcategory_averages.get("Connectivity and Synchronization", {}).get("Virtual-to-Physical Feedback", 0) < 2 and \
         subcategory_averages.get("Connectivity and Synchronization", {}).get("Synchronization", 0) < 2 and \
        category_averages.get("Modeling, Simulation and Decision Support", 0) > 3.5:
        classification = "Digital model"
        explanation = "Your system is a Digital model because it lacks both real-time synchronisation and feedback. it can be useful for situational decision-making but will need a lot of maintenance as it doesn't evolve or relate to the real world physical entity it represents."
        image_path = "images/digital_model.png"

    elif category_averages.get("Core Digital Twin Characteristics", 0) < 3 and \
         subcategory_averages.get("Connectivity and Synchronization", {}).get("Physical-to-Virtual Connection", 0)> 3 and \
         subcategory_averages.get("Connectivity and Synchronization", {}).get("Virtual-to-Physical Feedback", 0)> 3 and \
         subcategory_averages.get("Connectivity and Synchronization", {}).get("Synchronization", 0)> 3 and \
        category_averages.get("Modeling, Simulation and Decision Support", 0)< 2:
        classification = "Cyber-Physical System"
        explanation = "Your system is classified as a Cyber-Physical System because it focuses on integrating physical and digital components without full twin capabilities. the system has some digital representation but might not be fully detailed. when it comes to Connectivity and Synchronization, Relatively high scores are expected, especially if the system integrates well with sensors and data flows. Finally, lower scores are expected when it comes to modeling as the system doesn’t offer full simulation or proactive decision-making "
        image_path = "images/cyber_physical.jpg"

    elif scores["Core Digital Twin Characteristics"]["Physical Entity"][0] > 3 and \
         scores["Core Digital Twin Characteristics"]["Physical Entity"][1] > 3 and \
         scores["Core Digital Twin Characteristics"]["Virtual Entity"][0] > 3 and \
         scores["Core Digital Twin Characteristics"]["Virtual Entity"][1] > 3 and \
         subcategory_averages.get("Connectivity and Synchronization", {}).get("Physical-to-Virtual Connection", 0)< 2 and \
         subcategory_averages.get("Connectivity and Synchronization", {}).get("Virtual-to-Physical Feedback", 0)< 2 and \
         subcategory_averages.get("Connectivity and Synchronization", {}).get("Synchronization", 0)< 2  and \
        category_averages.get("Modeling, Simulation and Decision Support", 0)< 2:
        classification = "3D Models & CAD"
        explanation = "Your system is primarily a 3D Model or CAD representation, focusing on visualization rather than real-time integration."
        image_path = "images/3d_model.webp"

    elif category_averages.get("Data Management and Integration", 0) > 3 and \
         category_averages.get("Core Digital Twin Characteristics", 0) >  3 and \
         subcategory_averages.get("Connectivity and Synchronization", {}).get("Physical-to-Virtual Connection", 0) > 3 and \
         subcategory_averages.get("Connectivity and Synchronization", {}).get("Virtual-to-Physical Feedback", 0) < 2 and \
         subcategory_averages.get("Connectivity and Synchronization", {}).get("Synchronization", 0)< 3 and \
         category_averages.get("Modeling, Simulation and Decision Support", 0) < 2:
        classification = "Digital Thread"
        explanation = "Your system aligns with the concept of a Digital Thread, integrating lifecycle data but lacking simulation and autonomous decision-making."
        image_path = "images/digital_thread.jpeg"

    elif scores["Core Digital Twin Characteristics"]["Physical Entity"][0] >= 2 and \
         scores["Core Digital Twin Characteristics"]["Physical Entity"][1] >= 2 and \
         scores["Core Digital Twin Characteristics"]["Virtual Entity"][0] > 3 and \
         scores["Core Digital Twin Characteristics"]["Virtual Entity"][1] > 3 and \
         subcategory_averages.get("Connectivity and Synchronization", {}).get("Physical-to-Virtual Connection", 0)>3 and \
         subcategory_averages.get("Connectivity and Synchronization", {}).get("Virtual-to-Physical Feedback", 0)< 2 and \
         subcategory_averages.get("Connectivity and Synchronization", {}).get("Synchronization", 0)> 2  and \
        category_averages.get("Modeling, Simulation and Decision Support", 0)< 2:
        classification = "IoT or SCADA"
        explanation = "Your system fits into the IoT or SCADA category due to its strong reliance on data collection from sensors, without full Digital Twin intelligence. It lacks simulation modules, calculation engines or autonomy to interact with the physical entity."
        image_path = "images/iot_scada.png"

    else:
        classification = "Other / Not a Digital Twin"
        explanation = "Your system does not meet the core characteristics of a Digital Twin but may belong to another digital technology category."
        image_path = "images/other.png"

    return classification, explanation, image_path

# Define the evaluation framework with fuzzy categories

## Ajouter une question par rapport au niveau de synchronisation de l'outil vis à vis à l'application du jumeau numérique.
## Defenitly need to revisit this article for further questions : Digital Twins: A Maturity Model for Their Classification and Evaluation

evaluation_framework = {
    "Core Digital Twin Characteristics": {
        "description": "Digital Twins are virtual representations of physical entities, enabling seamless bi-directional data exchange for real-time monitoring, simulation, and decision-making. The core Digital Twin components are the physical entity, the virtual copy and the data transfer linking them to one another.",
        "subcategories": [
            {
                "subcategory": "Physical Entity",
                "questions": [
                    {"question": "How clearly is the real-world physical entity defined?", "type": "fuzzy"},
                    {"question": "How clearly are the entity’s boundaries and hierarchical levels specified regarding the purpose of the system?", "type": "fuzzy"},
                    {"question": "How well are physical and environmental parameters (e.g., temperature, pressure, operational context) influencing the physical system identified?", "type": "fuzzy"}
                ]
            },
            {
                "subcategory": "Virtual Entity",
                "questions": [
                    {"question": "How accurately does the virtual system represent the physical entity within its specific application context?", "type": "fuzzy"},
                    {"question": "Is the representation granular enough to capture detailed interactions and changes relevant to the system’s main objective (geometry, behavior, and functional rules)?", "type": "fuzzy"},
                    {"question": "Does the system include an intuitive user interface for interaction, access, analysis and experiment run?", "type": "fuzzy"}
                ]
            }
        ]
    },
    "Connectivity and Synchronization": {
        "description": "A fundamental feature of Digital Twins is their ability to maintain dynamic, bi-directional connections between physical and virtual entities. This involves ensuring synchronization through real-time or near-real-time data flows to support operational and strategic objectives, followed by the ability of the system to react and interact with the physical entity when needed (Whitin its application scope).",
        "subcategories": [
            {
                "subcategory": "Physical-to-Virtual Connection",
                "questions": [
                    {"question": "How automated is the process of transmitting data from the physical entity to the virtual system?", "type": "fuzzy"},
                    {"question": "How often is the data transmitted from the physical to the virtual system? (1= never and 5= real time)", "type": "fuzzy"},
                    {"question": "How well is the system integrated with other relevant systems (e.g., within a cloud environment, ERP, MES, IoT) (1= no integration and 5= autonomous communication with other systems)?", "type": "fuzzy"}
                ]
            },
            {
                "subcategory": "Virtual-to-Physical Feedback",
                "questions": [
                    {"question": "Is there a mechanism for real-time decision-making that optimizes physical operations?", "type": "fuzzy"},
                    {"question": "How seamlessly, within the scope of the system application, can it initiate  actions in the physical entity? (1= no reaction possible, 5= sending control commands or notifications to humans in the loop)", "type": "fuzzy"}
                ]
            },
            {
                "subcategory": "Synchronization",
                "questions": [
                    {"question": "Is the method for connecting the physical and virtual components (e.g., sensors, IoT devices, cloud computing) clearly defined?", "type": "fuzzy"},
                    {"question": "How well does the synchronization interval match the requirements for decision-making?", "type": "fuzzy"},
                    {"question": "How effectively does the system reflect historical, current, and predicted states of the physical entity?", "type": "fuzzy"}
                ]
            },
        ]
    },
    "Modeling, Simulation and Decision Support": {
        "description": "Digital Twins enable predictive and prescriptive capabilities through scomputational engines, providing actionable insights and optimization strategies. These capabilities align with transitioning decision-making from reactive to proactive processes.",
        "subcategories": [
            {
                "subcategory": "Modeling and What-If Scenarios",
                "questions": [
                    {"question": "Does the system have a computational engine to support simulation and decision-making?", "type": "fuzzy"},
                    {"question": "Can the system evaluate 'what-if' scenarios for varying operational settings?", "type": "fuzzy"}
                ]
            },
            {
                "subcategory": "Optimization and Decision Making",
                "questions": [
                    {"question": "can optimization algorithms be applied to improve performance metrics (e.g., logistics, costs, sustainability)?", "type": "fuzzy"},
                    {"question": "Can the system provide actionable insights to humans in the loop?", "type": "fuzzy"}
                ]
            }
        ]
    },
    "Data Management and Integration": {
        "description": "Digital Twins depend on robust data collection, integration, and processing frameworks to ensure seamless real-time operations. This includes IoT devices, cloud/edge computing, and compatibility with enterprise systems.",
        "subcategories": [
            {
                "subcategory": "System Integration",
                "questions": [
                    {"question": "How effectively does the system integrate multi-modal data (structured vs. unstructured, historical vs. real-time)?", "type": "fuzzy"},
                    {"question": "Can the system handle increasing data volumes and expanding functionalities seamlessly?", "type": "fuzzy"}
                ]
            },
            {
                "subcategory": "Data Fusion Collection and Processing",
                "questions": [
                    {"question": "How well can data heterogeneity be managed (handling different formats, resolutions, or sources)?", "type": "fuzzy"},
                    {"question": "To what extent are multiple data sources (sensor data, static data, predictive outputs) fused for comprehensive insights? ", "type": "fuzzy"}
                ]
            }
        ]
    },
    "Learning, Adaptability and Autonomy": {
        "description": "A mature Digital Twin leverages AI and machine learning to self-improve, recognize context changes, and adapt its models autonomously. This adaptability ensures scalability and relevance throughout its lifecycle.",
        "subcategories": [
            {
                "subcategory": "Context Awareness",
                "questions": [
                    {"question": "How dynamically does the system recognize environmental changes?", "type": "fuzzy"},
                    {"question": "How well does the system incorporate component interactions, disruptions, and uncertainties into its models?", "type": "fuzzy"}
                ]
            },
            {
                "subcategory": "Learning Capabilities",
                "questions": [
                    {"question": "How advanced is the system’s self-learning capability? (1 = No intelligence, 5 = Fully autonomous learning)", "type": "fuzzy"},
                    {"question": "To what extent does AI/ML contribute to predicting, analyzing, and optimizing performance?", "type": "fuzzy"},
                    {"question": "How interpretable and explainable are the system’s decisions?", "type": "fuzzy"}
                ]
            },
            {
                "subcategory": "Adaptability and Evolution",
                "questions": [
                    {"question": "How scalable is the system in integrating new equipment, functionalities, or processes?", "type": "fuzzy"},
                    {"question": "To what extent is the system applicable throughout its physical counterpart’s lifecycle?", "type": "fuzzy"},
                ]
            },
                        {
                "subcategory": "Autonomy",
                "questions": [
                    {"question": "How capable is the system of updating itself (e.g., its logic and parameters) without external intervention?", "type": "fuzzy"},
                    {"question": "How well can the system independently make and execute decisions within its predefined application scope?", "type": "fuzzy"}
                ]
            }
        ]
    },
    "Fidelity and Validation": {
        "description": "Digital Twins aim for high-fidelity representations while balancing computational efficiency. Validation ensures their trustworthiness and alignment with physical behaviors, critical for stakeholder confidence.",
        "subcategories": [
            {
                "subcategory": "Abstrction Level",
                "questions": [
                    {"question": "How well does the results of the system’s computational engine correspond to the actual behavior of the physical system (given the same stimuli)?", "type": "fuzzy"},
                    {"question": "How reproductible is the system’s behavior (given the same inputs)?", "type": "fuzzy"}
                ]
            },
            {
                "subcategory": "Verification and Feedback",
                "questions": [
                    {"question": "How comprehensively rigorous is the system’s verification process (e.g., testing, sensitivity analysis, real-world comparisons)?", "type": "fuzzy"},
                    {"question": "How effectively are real-world outcomes used for system adjustment and refinement?", "type": "fuzzy"}
                ]
            }
        ]
    },
    "Digital Twin Services": {
        "description": "The functional utility of Digital Twins is measured by their service capabilities, such as real-time monitoring, predictive maintenance, and operational optimization, aimed at enhancing the system's performance and resilience.",
        "subcategories": [
            {
                "subcategory": "Real-Time Monitoring",
                "questions": [
                    {"question": "How effectively does the system monitor key metrics (e.g., energy consumption, performance, errors) in real-time?", "type": "fuzzy"},
                    {"question": "How portable is the system across various devices and platforms?", "type": "fuzzy"}
                ]
            },
            {
                "subcategory": "Optimization and Forcasting",
                "questions": [
                    {"question": "How capable is the system of forecasting future states and emergency events?", "type": "fuzzy"},
                    {"question": "Does the system enable predictive analytics?", "type": "fuzzy"},
                    {"question": "Does the system provide prescriptive analytics?", "type": "fuzzy"}
                ]
            }
        ]
    },
    "Technological Readiness": {
        "description": "The deployment of Digital Twins depends on the integration of advanced technologies such as IoT, cloud computing, AI/ML, and ensuring cybersecurity. Scalability and compliance with privacy standards are also critical considerations.",
        "subcategories": [
            {
                "subcategory": "Enabling Technologies",
                "questions": [
                    {"question": "To what extent are advanced technologies (e.g., IoT, cloud/edge computing, AI/ML, big data, 5G) integrated into the system (regarding its application and scope)?", "type": "fuzzy"},
                    {"question": "Does the platform allow domain experts to operate the system without needing deep technical support expertise?", "type": "fuzzy"}
                ]
            },
            {
                "subcategory": "Security and Privacy",
                "questions": [
                    {"question": "To what extent does the system reliably prevent unauthorized access to data?", "type": "fuzzy"},
                    {"question": "How robust are the protective measures in place to guarantee data privacy?", "type": "fuzzy"}
                ]
            }
        ]
    }
}

# Ensure navigation state exists in session
if "navigation" not in st.session_state:
    st.session_state["navigation"] = "Profile Identification"  # Default page

# Initialize navigation state
if "redirect_to" not in st.session_state:
    st.session_state["redirect_to"] = None

# Sidebar navigation
page = st.sidebar.radio(
    "Select a Category",
    ["Profile Identification"] + list(evaluation_framework.keys()) + ["Summary", "Chatbot"],
    key="navigation"
)

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

    # Populate fields with existing session state values
    st.write("Before we dive in, let us know each other a little bit 🙂. The results of this questionnaire are to be collected for research purposes. It will take you approximately 15 minutes to complete. The results of the questionnaire will also allow you to interact with a Digital Twin literature GPT regarding your answers or interests. Feel free to add as many comments as needed to get concrete answers.")
    st.session_state["profile_data"]["field_of_work"] = st.radio(
        "What is your field of work?", 
        ["Research", "Industry", "logistics and supply chain"], 
        index=["Research", "Industry", "logistics and supply chain"].index(st.session_state["profile_data"].get("field_of_work", "Research"))
    )
    st.session_state["profile_data"]["years_experience"] = st.slider(
        "How many years have you worked on the systems you are testing with this survey?", 
        0, 50, step=1, 
        value=st.session_state["profile_data"].get("years_experience", 0)
    )
    st.session_state["profile_data"]["current_system"] = st.text_input(
        "What is the name of the system you use if any (Information System, Digital Twin, Warehouse Management System, IBM, Simulation, CPS ...)?", 
        value=st.session_state["profile_data"].get("current_system", "")
    )
    st.session_state["profile_data"]["position"] = st.text_input(
        "What is your current work position?", 
        value=st.session_state["profile_data"].get("position", "")
    )
    st.session_state["profile_data"]["country"] = st.text_input(
        "Which country are you based in?", 
        value=st.session_state["profile_data"].get("country", "")
    )
    st.session_state["profile_data"]["department"] = st.text_input(
        "What department are you affiliated with (e.g., R&D, Logistics)?", 
        value=st.session_state["profile_data"].get("department", "")
    )
    st.session_state["profile_data"]["comments"] = st.text_area(
        "Any additional comments or insights you would like to share?", 
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

# Validation function
def validate_all_answers():
    """Ensure all questions are answered by checking for None values."""
    for category, subcategories in st.session_state.scores.items():
        for subcat, answers in subcategories.items():
            if None in answers:  # Check for uninitialized (None) values
                return False, f"Please complete all questions in '{category}' - '{subcat}'."
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

# Main content rendering
st.title("Digital Twin Evaluation Framework")

if page == "Profile Identification":
    profile_data = profile_identification()
    st.session_state["profile_data"] = profile_data

elif page in evaluation_framework.keys():

    if st.session_state.scroll_to_top:
        scroll_to_here(0, key="top")  # Scroll to top
        st.session_state.scroll_to_top = False  # Reset state
    
    # Render evaluation questions
    st.subheader(f"Category: {page}")
    st.write(evaluation_framework[page]["description"])

    for subcategory in evaluation_framework[page]["subcategories"]:
        st.write(f"### {subcategory['subcategory']}")
        for i, question in enumerate(subcategory["questions"]):
            if st.session_state.scores[page][subcategory["subcategory"]][i] is None:
                st.session_state.scores[page][subcategory["subcategory"]][i] = 0
            st.session_state.scores[page][subcategory["subcategory"]][i] = st.slider(
                question["question"],  # Extract the question text
                0,
                5,
                step=1,
                key=f"{page}_{subcategory['subcategory']}_{i}",
                value=st.session_state.scores[page][subcategory["subcategory"]][i]
            )
        comment = st.text_area(
            f"Comments for {subcategory['subcategory']}",
            key=f"comment_{page}_{subcategory['subcategory']}",
            value=st.session_state.comments[page][subcategory["subcategory"]]
        )
        st.session_state.comments[page][subcategory["subcategory"]] = comment

elif page == "Summary":
    st.subheader("Evaluation Summary")

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

        st.write("""
            ### Ready to Proceed?
            The table above summarizes the evaluation scores for each category and subcategory.
            A radar graph below visualizes the average scores by category.
            Please click the 'Submit and Continue' button to share your results for further research. 😊
        """)

        # Prepare data for radar chart
        radar_data = summary_df.groupby("Category")["Average Score"].mean().tolist()
        radar_categories = summary_df["Category"].unique()

        # Plot radar chart
        plot_radar_chart(radar_data, radar_categories)

        # System classification
        st.markdown("---")
        st.subheader("System Classification")

        classification, explanation, image_path = classify_system(st.session_state.scores)

        st.subheader(f"System Classification: {classification}")

        # Display Image
        if os.path.exists(image_path):
            st.image(image_path, caption=classification,  use_container_width=True)
        else:
            st.warning(f"Image not found: {image_path}")

        # Display Explanation
        st.write(f"**Explanation:** {explanation}")

        # if the ssytem really is a Digital twin (in which case ... props to you!)
        if(classification == "Digital Twin"):
            # Display further explanation
            st.write(f"There is a lack of uniformity in the definition of digital twins, which further emphasizes the need for a standardized framework. The core characteristics of the technology are pretty straight forward. However, different maturity levels can still be identified in the Digital Twin paradigm itself. Here is a deeper analysis of the maturity of your disital twin:")

        # Data submission
        if st.button("Submit and Continue"):
            # Send data to Google Apps Script Web App
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
                        if subcategory_data:
                            question_text = subcategory_data["questions"][idx]["question"]
                        else:
                            raise ValueError(f"Subcategory '{subcat}' not found in category '{category}'.")

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

            # Send data to Google Sheets
            data_to_send = {
                "profile_data": profile_df,
                "scores": scores_df,
                "comments": comments_df,
                }

            # Convert dictionaries to a DataFrames
            profile_data_df = pd.DataFrame([profile_data])
            scores_data_df = pd.DataFrame(scores_data)
            comments_data_df = pd.DataFrame(comments_data)

            # Add the new data in the read datasets 
            updated_profiles = pd.concat([profile_df, profile_data_df], ignore_index=True)
            updated_scores = pd.concat([scores_df, scores_data_df], ignore_index=True)
            updated_comments = pd.concat([comments_df, comments_data_df], ignore_index=True)

            # Update Google Sheets with the new vendor data
            conn.update(worksheet="profile_data", data=updated_profiles)
            conn.update(worksheet="scores", data=updated_scores)
            conn.update(worksheet="comments", data=updated_comments)

            st.success("Your feedback has been successfully submitted!")
            st.success("Please head to the chatbot page on the left to discuss further with our custom GPT, trained on 57 research articles, based on your answers.")
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
