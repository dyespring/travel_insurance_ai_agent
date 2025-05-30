from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from flask_session import Session
from langchain_chroma import Chroma 
from components.recommendation.rec_system.recommend import recommend
from components.recommendation.rec_system.preprocessing import getDestination
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage
from datetime import datetime
import uuid
import os
import json
import shutil
from dotenv import load_dotenv
from datetime import timedelta
from werkzeug.utils import secure_filename

app = Flask(__name__, 
            template_folder='views',
            static_folder='views_style')

# Create session directory if it doesn't exist
os.makedirs('./flask_session', exist_ok=True)

# Enhanced session configuration
app.secret_key = os.getenv('SECRET_KEY', 'your-super-secret-key-change-in-production-12345') #
app.config['SESSION_TYPE'] = 'filesystem' #
app.config['SESSION_PERMANENT'] = True #
app.config['SESSION_USE_SIGNER'] = True #
app.config['SESSION_FILE_DIR'] = './flask_session' #
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24) #

# --> Add these lines for better cookie control in development <--
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # Or 'None', but 'None' typically requires Secure=True
app.config['SESSION_COOKIE_SECURE'] = False    # For HTTP development
app.config['SESSION_COOKIE_PATH'] = '/'        # Ensure cookie is valid for all paths
app.config['SESSION_COOKIE_DOMAIN'] = None

# Initialize Flask-Session
Session(app)

# Enhanced CORS configuration for proper session handling
CORS(app, 
     supports_credentials=True,
     origins=["http://localhost:8888", "http://127.0.0.1:8888", "http://localhost:3000", "http://127.0.0.1:3000"],
     allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     expose_headers=["Set-Cookie"],
     allow_credentials=True)

load_dotenv('.env')

# Global variables
vectorstore = None
retriever = None
chain = None
embeddings = None
llm = None
conversation_memory = {}  # Store conversations by session ID

def get_or_create_session_id():
    """Get or create a unique session ID for the user"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        session.permanent = True
        print(f"Created new session ID: {session['session_id']}")
    return session['session_id']

def get_conversation_memory(session_id):
    """Get or create conversation memory for a session"""
    if session_id not in conversation_memory:
        conversation_memory[session_id] = ConversationBufferWindowMemory(
            k=10,  # Keep last 10 exchanges (20 messages total)
            return_messages=True,
            memory_key="chat_history"
        )
        print(f"Created new conversation memory for session: {session_id}")
    return conversation_memory[session_id]

def cleanup_old_conversations():
    """Clean up old conversations to prevent memory buildup"""
    if len(conversation_memory) > 100:  # Arbitrary limit
        oldest_sessions = list(conversation_memory.keys())[:50]
        for session_id in oldest_sessions:
            del conversation_memory[session_id]

# Load datasets with error handling
def load_json_data():
    """Load all JSON data files with error handling"""
    data = {}
    
    files = {
        'insurance_plans': 'data/insurance_plans.json',
        'destination_risks': 'data/destination_risks.json',
        'activity_coverage': 'data/activity_coverage.json',
        'user_profiles': 'data/user_profiles.json'
    }
    
    for key, filepath in files.items():
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    data[key] = json.load(f)
                print(f"✓ Loaded {len(data[key])} items from {filepath}")
            else:
                print(f"✗ File not found: {filepath}")
                data[key] = []
        except Exception as e:
            print(f"✗ Error loading {filepath}: {e}")
            data[key] = []
    
    return data

# Load all data
json_data = load_json_data()
insurance_plans = json_data['insurance_plans']
destination_risks = json_data['destination_risks']
activity_coverage = json_data['activity_coverage'].get('activities', []) if isinstance(json_data['activity_coverage'], dict) else json_data['activity_coverage']
user_profiles = json_data['user_profiles']


def initialize_openai_components():
    """Initialize OpenAI components with error handling"""
    global embeddings, llm
    
    try:
        # Check if OpenAI API key is set
        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        embeddings = OpenAIEmbeddings()
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)  # Lower temperature for more consistent responses
        print("✓ OpenAI components initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Error initializing OpenAI components: {e}")
        return False

def create_vector_store(force_recreate=False):
    """Create or recreate the vector store with proper error handling"""
    global vectorstore, retriever
    
    print(f"\n=== Creating Vector Store (force_recreate={force_recreate}) ===")
    
    # Check if we have data to work with
    total_items = len(insurance_plans) + len(destination_risks) + len(activity_coverage)
    if total_items == 0:
        raise ValueError("No data available to create vector store")
    
    print(f"Available data: {len(insurance_plans)} insurance plans, {len(destination_risks)} destinations, {len(activity_coverage)} activities")
    
    # Remove existing database if force recreate
    if force_recreate and os.path.exists("data/chroma_db"):
        print("Removing existing vector store...")
        shutil.rmtree("data/chroma_db")
    
    # Create directory
    os.makedirs("data/chroma_db", exist_ok=True)
    
    # Prepare documents
    documents = []
    
    # Add insurance plans
    print("Processing insurance plans...")
    for i, plan in enumerate(insurance_plans):
        try:
            text = f"Insurance Plan: {plan.get('name', 'Unknown')} by {plan.get('insurer', 'Unknown')}. "
            text += f"Coverage: {plan.get('coverage', 'Not specified')}. "
            activities = plan.get('activities', [])
            if activities:
                text += f"Activities covered: {', '.join(activities)}. "
            text += f"Plan ID: {plan.get('plan_id', 'Unknown')}"
            
            doc = Document(
                page_content=text,
                metadata={"type": "insurance_plan", "id": str(plan.get("plan_id", "unknown"))}
            )
            documents.append(doc)
            print(f"  ✓ Plan {i+1}: {text[:50]}...")
        except Exception as e:
            print(f"  ✗ Error processing insurance plan {i+1}: {e}")
    
    # Add destination risks
    print("Processing destinations...")
    for i, dest in enumerate(destination_risks):
        try:
            text = f"Destination: {dest.get('country', 'Unknown')} ({dest.get('iso_code', 'Unknown')}). "
            text += f"Risk Level: {dest.get('risk_level', 'Unknown')}. "
            text += f"COVID Requirements: {dest.get('covid_requirements', 'Not specified')}"
            
            doc = Document(
                page_content=text,
                metadata={"type": "destination", "id": str(dest.get("iso_code", "unknown"))}
            )
            documents.append(doc)
            print(f"  ✓ Destination {i+1}: {text[:50]}...")
        except Exception as e:
            print(f"  ✗ Error processing destination {i+1}: {e}")
    
    # Add activities
    print("Processing activities...")
    for i, activity in enumerate(activity_coverage):
        try:
            text = f"Activity: {activity.get('activity', 'Unknown')}. "
            text += f"Coverage Level: {activity.get('coverage_level', 'Unknown')}. "
            cost = activity.get('rider_cost', 'N/A')
            text += f"Additional Cost: {cost}"
            
            doc = Document(
                page_content=text,
                metadata={"type": "activity", "id": activity.get("activity", "unknown").lower().replace(" ", "_")}
            )
            documents.append(doc)
            print(f"  ✓ Activity {i+1}: {text[:50]}...")
        except Exception as e:
            print(f"  ✗ Error processing activity {i+1}: {e}")
    
    print(f"Created {len(documents)} documents total")
    
    if len(documents) == 0:
        raise ValueError("No valid documents created")
    
    # Create vector store
    try:
        print("Creating Chroma vector store...")
        
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory="data/chroma_db"
        )
        
        print("Vector store created, checking document count...")
        doc_count = vectorstore._collection.count()
        print(f"Vector store reports {doc_count} documents")
        
        if doc_count == 0:
            print("WARNING: Vector store created but contains 0 documents!")
            # Try to add documents manually
            vectorstore.add_documents([documents[0]])
            new_count = vectorstore._collection.count()
            print(f"After manual add_documents: {new_count} documents")
            
            if new_count > 0:
                print("Manual add worked, trying to add all documents...")
                vectorstore.add_documents(documents[1:])
                final_count = vectorstore._collection.count()
                print(f"Final count after manual adds: {final_count} documents")
            else:
                raise ValueError("Cannot add documents to vector store - possible embedding issue")
        
        # Create retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        print("✓ Retriever created")
        
        # Final verification
        final_count = vectorstore._collection.count()
        print(f"✓ Vector store creation complete with {final_count} documents")
        
        return vectorstore
        
    except Exception as e:
        print(f"✗ Error creating vector store: {e}")
        import traceback
        traceback.print_exc()
        raise

def is_out_of_scope(question, retrieved_context, threshold=0.3):
    """
    Determine if question is out of scope
    """
    # Case 1: No context retrieved or empty documents
    if not retrieved_context or all(not doc.page_content.strip() for doc in retrieved_context):
        return True
    
    # Case 2: Check similarity scores if available
    if hasattr(retrieved_context[0], 'metadata') and 'score' in retrieved_context[0].metadata:
        if retrieved_context[0].metadata['score'] < threshold:
            return True
    
    # Case 3: Domain keyword check
    domain_keywords = {
        "insurance", "travel", "coverage", "plan", 
        "destination", "risk", "activity", "medical",
        "trip", "policy", "evacuation", "cancel"
    }
    question_words = set(question.lower().split())
    return len(question_words & domain_keywords) == 0


# Add this function near the other utility functions (around line 200)
def get_dynamic_prompt_template(conversation_context=None):
    """Generate a dynamic prompt template based on conversation context"""
    
    # Default template for general questions
    base_template = """You are a helpful and knowledgeable travel insurance expert. Your goal is to provide accurate, personalized advice to help users make informed decisions about their travel insurance needs.

Context from knowledge base:
{context}

User Profile (if available):
{user_context}

Previous conversation history:
{chat_history}

Current question: {question}"""

    # Dynamic sections based on context
    dynamic_sections = {
        'insurance_recommendation': """
Instructions for insurance recommendations:
1. Analyze the user's profile and trip details carefully
2. Recommend specific plans that match their needs
3. Clearly explain why each plan is suitable
4. Be transparent about coverage limitations
5. Format recommendations clearly with sections
6. Be specific about what is and isn't covered
7. If you don't have specific information, say so clearly

📌 **Plan: [Plan Name] by [Insurer]**
💰 **Price:** [Estimated Cost or "Contact for quote"]
🛡️ **Coverage Includes:** [Bullet points]
🌟 **Why We Recommend This:** [Explanation]
🔹 **Best For:** [Traveler type/situation]
🔹 **Additional Notes:** [Important details]""",

        'destination_risk': """
Instructions for destination risk questions:
1. Provide up-to-date risk assessment
2. Include any travel advisories
3. Mention COVID requirements if relevant
4. Suggest appropriate insurance coverage
5. Format with clear sections

🌍 **Destination:** [Country Name]
⚠️ **Risk Level:** [Level and explanation]
🛡️ **Recommended Coverage:** [Types needed]
📝 **Requirements:** [Visa, vaccines, etc.]""",

        'activity_coverage': """
Instructions for activity coverage questions:
1. Specify if the activity is covered
2. Mention any special riders needed
3. Provide cost estimates if available
4. Suggest alternative activities if risky
5. Format with clear sections

🎯 **Activity:** [Activity Name]
✅ **Coverage Status:** [Covered/Not Covered]
💲 **Additional Cost:** [If applicable]
📋 **Requirements:** [Safety measures, etc.]""",

        'general_advice': """
Instructions for general advice:
1. Provide clear, actionable information
2. Reference official sources when possible
3. Tailor advice to user's profile
4. Suggest next steps if appropriate"""
    }

    # Determine which sections to include
    selected_sections = [base_template]
    
    if conversation_context:
        # Add specific sections based on detected context
        if 'insurance' in conversation_context.lower():
            selected_sections.append(dynamic_sections['insurance_recommendation'])
        if 'destination' in conversation_context.lower() or 'country' in conversation_context.lower():
            selected_sections.append(dynamic_sections['destination_risk'])
        if 'activity' in conversation_context.lower() or 'sport' in conversation_context.lower():
            selected_sections.append(dynamic_sections['activity_coverage'])
    
    # Always include general advice at the end
    selected_sections.append(dynamic_sections['general_advice'])
    
    # Add final instructions
    selected_sections.append("""
Final Instructions:
1. NEVER start responses with empty lines or spaces
2. Be professional yet approachable
3. Use emojis sparingly for readability
4. Keep responses concise but complete
5. Offer to help with follow-up questions
6. Always prioritize user safety
7. If you don't have specific information, say so clearly""")

    return "\n\n".join(selected_sections)

# Update the setup_rag_chain_with_memory function (around line 250)
def setup_rag_chain_with_memory():
    """Setup the RAG chain with conversation memory"""
    global chain
    
    # Define the dynamic template selection logic separately
    def determine_template_type(inputs):
        """Analyze inputs to determine template type"""
        question = inputs.get("question", "").lower()
        chat_history = inputs.get("chat_history", "")
        
        if any(word in question for word in ["plan", "coverage", "insurance"]):
            return "insurance_recommendation"
        elif any(word in question for word in ["country", "destination", "travel to","requirements","risks","visa"]):
            return "destination_risk"
        elif any(word in question for word in ["activity", "sport", "hiking","diving","ski"]):
            return "activity_coverage"
        return "general_advice"

    # Create static prompt templates for each type
    template_mappings = {
        "insurance_recommendation": get_dynamic_prompt_template("insurance"),
        "destination_risk": get_dynamic_prompt_template("destination"),
        "activity_coverage": get_dynamic_prompt_template("activity"),
        "general_advice": get_dynamic_prompt_template()
    }

    # Main prompt template that selects the right sub-template
    full_template = """You are a travel insurance expert. Answer based on:
    
Context: {context}
User Profile: {user_context}
Chat History: {chat_history}

{template_selection}

Question: {question}"""

    prompt = ChatPromptTemplate.from_template(full_template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs) if docs else "No context found"

    chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(x["question"])),
            "user_context": lambda x: x["user_context"],
            "chat_history": lambda x: x.get("chat_history", ""),
            "template_selection": lambda x: template_mappings[determine_template_type(x)],
            "question": lambda x: x["question"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )
def format_response(text_response):
    """Clean up response formatting"""
    if not isinstance(text_response, str):
        text_response = str(text_response)
    
    # First, strip all leading and trailing whitespace
    text_response = text_response.strip()

        # Remove empty lines at start
    while text_response.startswith('\n'):
        text_response = text_response[1:]
    
    # Ensure proper spacing around sections
    sections = [
        ("📌 **Plan:", "\n📌 **Plan:"),
        ("💰 **Price:**", "\n💰 **Price:**"),
        ("🛡️ **Coverage Includes:**", "\n🛡️ **Coverage Includes:**"),
        ("🌟 **Why We Recommend This:**", "\n🌟 **Why We Recommend This:**"),
        ("🔹 **Best For:**", "\n🔹 **Best For:**"),
        ("🔹 **Additional Notes:**", "\n🔹 **Additional Notes:**"),
        ("---", "\n---")
    ]
    
    for original, replacement in sections:
        text_response = text_response.replace(original, replacement)

    # Remove any double newlines and replace with single newlines
    while '\n\n' in text_response:
        text_response = text_response.replace('\n\n', '\n')
    
    # If the response starts with a newline + emoji, remove the leading newline
    if text_response.startswith('\n📌'):
        text_response = text_response[1:]

    return text_response.strip()

def initialize_system():
    """Initialize the entire system with proper error handling"""
    global vectorstore, retriever, chain
    
    print("=== Initializing Travel Insurance Assistant ===")

    # 1. Initialize OpenAI components
    if not initialize_openai_components():
        raise RuntimeError("Failed to initialize OpenAI components")
    
    # 2. Try to load existing vector store
    try:
        if os.path.exists("data/chroma_db"):
            print("Attempting to load existing vector store...")
            vectorstore = Chroma(
                persist_directory="data/chroma_db",
                embedding_function=embeddings
            )
            
            doc_count = vectorstore._collection.count()
            print(f"Loaded existing vector store with {doc_count} documents")
            
            if doc_count == 0:
                print("Existing vector store is empty, recreating...")
                raise ValueError("Empty vector store")
            
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            print("✓ Retriever created from existing vector store")
            
        else:
            raise FileNotFoundError("Vector store directory doesn't exist")
            
    except Exception as e:
        print(f"Failed to load existing vector store: {e}")
        print("Creating new vector store...")
        vectorstore = create_vector_store(force_recreate=True)
    
    # 3. Ensure retriever is set
    if not retriever and vectorstore:
        print("Retriever not set, creating from vectorstore...")
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        print("✓ Retriever created")
    
    # 4. Setup RAG chain
    if retriever:
        setup_rag_chain_with_memory()
    else:
        print("✗ Cannot setup RAG chain - no retriever available")
        raise RuntimeError("Failed to create retriever")
    
    print("✓ System initialization complete!\n")

def query_rag_system(question, user_context=None, session_id=None):
    """Query the RAG system with out-of-scope detection"""
    if not chain or not retriever:
        raise ValueError("RAG chain or retriever not initialized")
    
    try:
        # First retrieve context to check relevance
        retrieved_context = retriever.invoke(question)  # Changed variable name for clarity
        
        # Out-of-scope detection
        if is_out_of_scope(question, retrieved_context):
            return format_response(
                "I specialize in travel insurance questions. "
                "I couldn't find relevant information for your query. "
                "Please ask about travel insurance plans, destination risks, "
                "or activity coverage."
            )
        
        # Proceed with normal RAG flow if in-scope
        if session_id:
            memory = get_conversation_memory(session_id)
            chat_history = memory.chat_memory.messages
        else:
            chat_history = []
        
        chain_input = {
            "question": question,
            "user_context": user_context or "No user profile available",
            "chat_history": chat_history,
            "context": "\n\n".join([doc.page_content for doc in retrieved_context])  # Add this line
        }
        
        answer = chain.invoke(chain_input)
        
        if session_id:
            memory.chat_memory.add_user_message(question)
            memory.chat_memory.add_ai_message(answer)
        
        return format_response(answer)
    
    except Exception as e:
        print(f"Error processing query: {e}")
        import traceback
        traceback.print_exc()
        raise

def get_user_context_string(user):
    """Convert user object to formatted context string"""
    if not user:
        return "No user profile available"
    
    context = f"""
User Profile:
- Name: {user.get('name', 'Unknown')}
- Age: {user.get('age', 'Not specified')}
- Traveler Type: {user.get('traveler_type', 'Not specified')}
- Upcoming Trip Destination: {user.get('upcoming_trip', {}).get('destination', 'Not specified')}
- Trip Duration: {user.get('upcoming_trip', {}).get('duration', 'Not specified')}
- Insurance Priority: {user.get('insurance_preferences', {}).get('priority', 'Not specified')}
- Risk Profile: {user.get('risk_profile', 'Not specified')}
"""
    return context.strip()

# Initialize the system
try:
    initialize_system()
    print("🚀 Travel Insurance Assistant is ready!")
except Exception as e:
    print(f"CRITICAL ERROR: System initialization failed: {e}")
    print("The application may not function properly.")

# Flask routes
@app.route('/')
def home():
    """Serve the main chatbot page"""
    return render_template("chatbot_final.html")

@app.route('/api/login', methods=['POST'])
def handle_login():
    """Handle user login with enhanced validation and session management"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        first_name = data.get('first_name', '').strip()
        last_name = data.get('last_name', '').strip()
        full_name = data.get('full_name', f"{first_name} {last_name}").strip()
        
        if not first_name or not last_name:
            return jsonify({"error": "First name and last name are required"}), 400
        
        print(f"Login attempt for: {full_name}")
        
        # Try to find existing user in profiles
        user = next((u for u in user_profiles if u['name'].lower() == full_name.lower()), None)
        
        if not user:
            # Create a temporary user profile for demo purposes
            user = {
                "user_id": f"TEMP-{len(user_profiles) + 1}",
                "name": full_name,
                "age": 30,
                "traveler_type": "Leisure",
                "upcoming_trip": {
                    "destination": "Not specified",
                    "duration": "Not specified"
                },
                "insurance_preferences": {
                    "priority": "Comprehensive coverage"
                },
                "risk_profile": "Moderate"
            }
            print(f"Created temporary user profile for: {full_name}")
        else:
            print(f"Found existing user profile for: {full_name}")
        
        # Get or create session ID
        session_id = get_or_create_session_id()
        
        # Store user info in server session
        session['current_user'] = user
        session.permanent = True
        
        print(f"User logged in successfully: {user['name']} (Session: {session_id})")
        print(f"Session contents: {dict(session)}")
        
        return jsonify({
            "status": "success", 
            "user": user,
            "message": f"Welcome, {first_name}!",
            "session_id": session_id
        })
        
    except Exception as e:
        print(f"Login error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Login failed. Please try again."}), 500

@app.route('/api/logout', methods=['POST'])
def handle_logout():
    """Handle user logout"""
    try:
        session_id = session.get('session_id')
        user_name = session.get('current_user', {}).get('name', 'Unknown')
        
        # Clear session data
        session.clear()
        
        # Clean up conversation memory
        if session_id and session_id in conversation_memory:
            del conversation_memory[session_id]
        
        print(f"User logged out: {user_name} (Session: {session_id} cleared)")
        return jsonify({"status": "success", "message": "Logged out successfully"})
    except Exception as e:
        print(f"Logout error: {e}")
        return jsonify({"error": "Logout failed"}), 500

@app.route('/api/chat', methods=['POST'])
def chatbot():
    """Enhanced chatbot endpoint with proper session handling"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        question = data.get('question', '').strip()
        if not question:
            return jsonify({"error": "No question provided"}), 400
        
        # Get session ID and user context from session
        session_id = get_or_create_session_id()
        current_user = session.get('current_user')
        user_context = get_user_context_string(current_user)
        
        if current_user:
            print(f"Processing query for user: {current_user.get('name', 'Unknown')} (Session: {session_id})")
        else:
            print(f"Processing query for anonymous user (Session: {session_id})")
        
        print(f"Session contents: {dict(session)}")
        
        # Query the RAG system with session ID for memory
        answer = query_rag_system(question, user_context, session_id)
        formatted_response = format_response(answer)
        
        return jsonify({
            "answer": formatted_response,
            "user_logged_in": current_user is not None,
            "user_name": current_user.get('name') if current_user else None,
            "session_id": session_id,
            "status": "success"
        })
        
    except Exception as e:
        print(f"Chat error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": f"Query failed: {str(e)}",
            "status": "error"
        }), 500

@app.route("/recommend")
def recommend_input():
    """Serve recommendation input page"""
    try:
        options = getDestination()
        return render_template("rec_input.html", options=options)
    except Exception as e:
        print(f"Error loading recommendation page: {e}")
        return f"Error loading recommendation page: {e}", 500

@app.route("/predict", methods=["POST"])
def recommend_output():
    """Handle recommendation predictions"""
    try:
        destination = request.form.get("destination")
        duration = request.form.get("duration")
        gender = request.form.get("gender")
        age = request.form.get("age")
       
        result = recommend(gender, destination, duration, age)

        # Load reviews
        try:
            with open("reviews.json", "r") as f:
                review_data = json.load(f)
        except:
            review_data = []

        review_lookup = {entry["company_name"]: entry for entry in review_data}
        for r in result:
            agency = r.get("Agency")
            review = review_lookup.get(agency)
            if review:
                r.update({
                    "average_rating": review["average_rating"],
                    "positive": review["positive"],
                    "neutral": review["neutral"],
                    "negative": review["negative"],
                    "summary": review["summary"]
                })
            else:
                r.update({
                    "average_rating": "N/A",
                    "positive": 0,
                    "neutral": 0,
                    "negative": 0,
                    "summary": "No reviews available."
                })
        
        return render_template('rec_output.html', result=result)
    except Exception as e:
        print(f"Error generating recommendations: {e}")
        return f"Error generating recommendations: {e}", 500

@app.route('/api/debug/recreate_vectorstore', methods=['POST'])
def recreate_vectorstore():
    """Force recreate the vector store for debugging"""
    try:
        global vectorstore, retriever, chain
        
        if not embeddings:
            return jsonify({"error": "OpenAI components not initialized"}), 500
        
        print("Debug: Recreating vector store...")
        vectorstore = create_vector_store(force_recreate=True)
        setup_rag_chain_with_memory()
        
        return jsonify({
            "status": "success",
            "message": "Vector store recreated successfully",
            "document_count": vectorstore._collection.count()
        })
    except Exception as e:
        print(f"Error recreating vector store: {e}")
        return jsonify({"error": str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    print("🚀 Starting Travel Insurance Assistant...")
    app.run(host='0.0.0.0', port=8888, debug=True)