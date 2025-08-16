import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import numpy as np
from recommendation import cnv, dme, drusen, normal
import tempfile

# Page configuration
st.set_page_config(
    page_title="OCT Retinal Analysis Platform",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Calendly-inspired modern theme
st.markdown("""
<style>
    /* Main theme colors - Calendly inspired */
    :root {
        --primary-blue: #006bff;
        --secondary-blue: #0056cc;
        --accent-green: #00d4aa;
        --light-green: #e6f7f3;
        --dark-blue: #1a1a2e;
        --light-blue: #f8faff;
        --white: #ffffff;
        --gray: #f8f9fa;
        --light-gray: #f1f3f4;
        --dark-gray: #5f6368;
        --border-gray: #e8eaed;
        --text-primary: #202124;
        --text-secondary: #5f6368;
    }
    
    /* Global styles */
    .main {
        background: var(--white);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Header styling - Calendly style */
    .main-header {
        background: transparent;
        padding: 3rem 2rem;
        margin: 2rem 0 3rem 0;
        text-align: center;
        position: relative;
    }
    
    .main-header h1 {
        color: var(--primary-blue);
        font-size: 3.5rem;
        font-weight: 700;
        margin: 0;
        text-align: center;
        letter-spacing: -0.02em;
        position: relative;
        z-index: 1;
    }
    
    .main-header p {
        color: var(--text-secondary);
        font-size: 1.3rem;
        text-align: center;
        margin: 1rem 0 0 0;
        opacity: 0.95;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }
    
    /* Fluid card styling - Calendly inspired */
    .feature-card {
        background: var(--white);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
        border: 1px solid var(--border-gray);
        margin-bottom: 2rem;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--primary-blue), var(--accent-green));
        transform: scaleX(0);
        transition: transform 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.12);
        border-color: var(--primary-blue);
    }
    
    .feature-card:hover::before {
        transform: scaleX(1);
    }
    
    .feature-card h3 {
        color: var(--text-primary);
        margin-bottom: 1rem;
        font-size: 1.6rem;
        font-weight: 600;
        letter-spacing: -0.01em;
    }
    
    .feature-card p {
        color: var(--text-secondary);
        line-height: 1.6;
        font-size: 1rem;
    }
    
    /* Disease cards - more organic */
    .disease-card {
        background: var(--white);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
        margin-bottom: 2rem;
        border: 1px solid var(--border-gray);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
    }
    
    .disease-card::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--accent-green), var(--primary-blue));
        transform: scaleX(0);
        transition: transform 0.3s ease;
    }
    
    .disease-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 16px 48px rgba(0, 0, 0, 0.12);
        border-color: var(--accent-green);
    }
    
    .disease-card:hover::after {
        transform: scaleX(1);
    }
    
    .disease-card h4 {
        color: var(--text-primary);
        margin-bottom: 1rem;
        font-size: 1.4rem;
        font-weight: 600;
    }
    
    .disease-card p {
        color: var(--text-secondary);
        line-height: 1.6;
    }
    
    /* Button styling - Calendly style */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-blue) 0%, var(--secondary-blue) 100%);
        color: var(--white);
        border: none;
        padding: 1rem 2.5rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 8px 24px rgba(0, 107, 255, 0.3);
        letter-spacing: 0.01em;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 32px rgba(0, 107, 255, 0.4);
        background: linear-gradient(135deg, var(--secondary-blue) 0%, var(--primary-blue) 100%);
    }
    
    /* File uploader styling */
    .stUploadedFile {
        background: var(--light-blue);
        border: 2px dashed var(--primary-blue);
        border-radius: 20px;
        padding: 2.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stUploadedFile:hover {
        border-color: var(--accent-green);
        background: var(--light-green);
    }
    
    /* Success message styling */
    .stSuccess {
        background: var(--accent-green);
        color: var(--white);
        padding: 1.5rem 2rem;
        border-radius: 20px;
        font-weight: 600;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 24px rgba(0, 212, 170, 0.2);
        border: none;
        font-size: 1.1rem;
        letter-spacing: 0.01em;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: var(--white);
        border-right: 1px solid var(--border-gray);
    }
    
    .sidebar .sidebar-content {
        background: var(--white);
    }
    
    /* Sidebar text styling */
    .css-1d391kg .stSelectbox label,
    .css-1d391kg .stSelectbox select,
    .css-1d391kg .stSelectbox option {
        color: var(--text-primary) !important;
        font-weight: 600;
    }
    
    /* Override Streamlit's default sidebar text colors */
    .css-1d391kg .stSelectbox > div > div {
        background-color: var(--white);
        border: 1px solid var(--border-gray);
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    .css-1d391kg .stSelectbox > div > div:hover {
        border-color: var(--primary-blue);
        box-shadow: 0 4px 12px rgba(0, 107, 255, 0.1);
    }
    
    /* Sidebar title styling */
    .css-1d391kg h2 {
        color: var(--primary-blue) !important;
        font-weight: 700;
        letter-spacing: -0.01em;
    }
    
    /* Stats cards - fluid design */
    .stats-container {
        display: flex;
        justify-content: space-between;
        margin: 3rem 0;
        flex-wrap: wrap;
        gap: 1.5rem;
    }
    
    .stat-card {
        background: var(--white);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
        text-align: center;
        flex: 1;
        min-width: 200px;
        border: 1px solid var(--border-gray);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .stat-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--primary-blue), var(--accent-green));
    }
    
    .stat-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 16px 48px rgba(0, 0, 0, 0.12);
    }
    
    .stat-number {
        font-size: 3rem;
        font-weight: 700;
        color: var(--primary-blue);
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .stat-label {
        color: var(--text-secondary);
        font-size: 1rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Section headers */
    .section-header {
        text-align: center;
        margin: 4rem 0 2rem 0;
        position: relative;
    }
    
    .section-header h2 {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 1rem;
        letter-spacing: -0.02em;
    }
    
    .section-header::after {
        content: '';
        position: absolute;
        bottom: -10px;
        left: 50%;
        transform: translateX(-50%);
        width: 60px;
        height: 3px;
        background: linear-gradient(90deg, var(--primary-blue), var(--accent-green));
        border-radius: 2px;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .stats-container {
            flex-direction: column;
        }
        .stat-card {
            margin: 0.5rem 0;
        }
        .main-header h1 {
            font-size: 2.5rem;
        }
        .main-header p {
            font-size: 1.1rem;
        }
    }
    
    /* Smooth scrolling */
    html {
        scroll-behavior: smooth;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--light-gray);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary-blue);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--secondary-blue);
    }
</style>
""", unsafe_allow_html=True)

# Tensorflow Model Prediction
def model_prediction(test_image_path):
    try:
        model = tf.keras.models.load_model("Trained_Eye_disease_model.keras")
        img = tf.keras.utils.load_img(test_image_path, target_size=(224, 224))
        x = tf.keras.utils.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        predictions = model.predict(x)
        return np.argmax(predictions)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <h2 style="color: #006bff; margin-bottom: 2rem; font-weight: 600; letter-spacing: -0.01em;">OCT Platform</h2>
    </div>
    """, unsafe_allow_html=True)
    
    app_mode = st.selectbox(
        "Select Page",
        ["Home", "About", "Disease Identification"],
        format_func=lambda x: x
    )

# Main Page
if app_mode == "Home":
    # Hero Section
    st.markdown("""
    <div class="main-header">
        <h1>OCT Retinal Analysis Platform</h1>
        <p>Advanced deep learning analysis powered by TensorFlow for retinal OCT imaging</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features Section
    st.markdown('<div class="section-header"><h2>Key Features</h2></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>Automated Image Analysis</h3>
            <p>State-of-the-art machine learning models classify OCT images into distinct categories: Normal, CNV, DME, and Drusen with high accuracy.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>Cross-Sectional Imaging</h3>
            <p>Examine high-quality images showcasing both normal retinas and various pathologies, helping doctors make informed clinical decisions.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>Streamlined Workflow</h3>
            <p>Upload, analyze, and review OCT scans in a few easy steps with an intuitive interface designed for medical professionals.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>Expert Verification</h3>
            <p>Multi-tier grading system with ophthalmologists and retinal specialists ensures the highest quality dataset and results.</p>
        </div>
        """, unsafe_allow_html=True)
    
 
    
    # Get Started Section
    st.markdown('<div class="section-header"><h2>Get Started</h2></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <h3>Ready to analyze OCT images?</h3>
        <p>Navigate to the <strong>Disease Identification</strong> page to upload and analyze retinal OCT scans using the advanced deep learning model.</p>
    </div>
    """, unsafe_allow_html=True)

# About Project
elif app_mode == "About":
    st.markdown("""
    <div class="main-header">
        <h1>About the Project</h1>
        <p>Learn about our dataset, methodology, and validation process</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Project Overview
    st.markdown('<div class="section-header"><h2>Project Overview</h2></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <h3>About Dataset</h3>
        <p>Retinal optical coherence tomography (OCT) is an imaging technique used to capture high-resolution cross sections of the retinas of living patients. Approximately 30 million OCT scans are performed each year, and the analysis and interpretation of these images takes up a significant amount of time.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <h3>Dataset Content</h3>
        <p>The dataset consists of <strong>84,495 high-resolution OCT images</strong> (JPEG format) organized into <strong>train, test, and validation</strong> sets, split into four primary categories:</p>
        <ul>
            <li><strong>Normal</strong></li>
            <li><strong>CNV</strong></li>
            <li><strong>DME</strong></li>
            <li><strong>Drusen</strong></li>
        </ul>
        <p>Images are labeled as (disease)-(randomized patient ID)-(image number by this patient) and split into 4 directories: CNV, DME, DRUSEN, and NORMAL.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <h3>Data Collection & Sources</h3>
        <p>Optical coherence tomography (OCT) images (Spectralis OCT, Heidelberg Engineering, Germany) were selected from retrospective cohorts of adult patients from the Shiley Eye Institute of the University of California San Diego, the California Retinal Research Foundation, Medical Center Ophthalmology Associates, the Shanghai First People's Hospital, and Beijing Tongren Eye Center between July 1, 2013 and March 1, 2017.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Clinical Significance
    st.markdown('<div class="section-header"><h2>Clinical Significance</h2></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="disease-card">
            <h4>Early Detection</h4>
            <p>OCT enables detection of retinal abnormalities before they cause noticeable symptoms, allowing for early intervention and better treatment outcomes.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="disease-card">
            <h4>Treatment Monitoring</h4>
            <p>Regular OCT scans help monitor treatment effectiveness and disease progression, enabling timely adjustments to therapy.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="disease-card">
            <h4>Non-Invasive</h4>
            <p>Unlike traditional diagnostic methods, OCT is completely non-invasive and painless, making it ideal for regular screening.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="disease-card">
            <h4>High Resolution</h4>
            <p>OCT provides micron-level resolution, revealing details that are invisible with other imaging techniques.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Disease Examples
    st.markdown('<div class="section-header"><h2>Understanding Retinal Diseases</h2></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="disease-card">
            <h4>CNV (Choroidal Neovascularization)</h4>
            <p>Far left: choroidal neovascularization with neovascular membrane (white arrowheads) and associated subretinal fluid (arrows). CNV is a serious complication of age-related macular degeneration where abnormal blood vessels grow under the retina, causing rapid vision loss.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="disease-card">
            <h4>DME (Diabetic Macular Edema)</h4>
            <p>Middle left: Diabetic macular edema with retinal-thickening-associated intraretinal fluid (arrows). DME occurs when fluid accumulates in the macula due to diabetic retinopathy, causing central vision blurring and distortion.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="disease-card">
            <h4>Drusen (Early AMD)</h4>
            <p>Middle right: Multiple drusen (arrowheads) present in early AMD. Drusen are yellow deposits under the retina that indicate early stages of age-related macular degeneration, a leading cause of vision loss in older adults.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="disease-card">
            <h4>Normal Retina</h4>
            <p>Far right: Normal retina with preserved foveal contour and absence of any retinal fluid/edema. A healthy retina shows clear layers, no fluid accumulation, and a well-defined foveal depression.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Dataset Details
    st.markdown('<div class="section-header"><h2>Dataset & Methodology</h2></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <h3>Dataset Composition</h3>
        <p>Our comprehensive dataset consists of <strong>84,495 high-quality OCT images</strong> collected from multiple prestigious medical institutions worldwide. The images are organized into three main subsets:</p>
        <ul>
            <li><strong>Training Set:</strong> 67,596 images for model development and optimization</li>
            <li><strong>Validation Set:</strong> 11,025 images for hyperparameter tuning and model selection</li>
            <li><strong>Test Set:</strong> 5,874 images for final performance evaluation</li>
        </ul>
        <p>Each subset maintains the same class distribution to ensure unbiased evaluation.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <h3>Data Collection & Sources</h3>
        <p>The OCT images were collected from multiple renowned medical centers between July 1, 2013, and March 1, 2017:</p>
        <ul>
            <li><strong>Shiley Eye Institute</strong> - University of California San Diego</li>
            <li><strong>California Retinal Research Foundation</strong></li>
            <li><strong>Medical Center Ophthalmology Associates</strong></li>
            <li><strong>Shanghai First People's Hospital</strong></li>
            <li><strong>Beijing Tongren Eye Center</strong></li>
        </ul>
        <p>All images were acquired using Spectralis OCT (Heidelberg Engineering, Germany) with standardized imaging protocols.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Validation Process
    st.markdown('<div class="section-header"><h2>Quality Assurance & Validation</h2></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <h3>Multi-Tier Grading System</h3>
        <p>Each image underwent a rigorous, three-tier validation process to ensure the highest quality and accuracy:</p>
        <ul>
            <li><strong>Tier 1 - Initial Screening:</strong> Undergraduate and medical students with specialized OCT interpretation training conducted initial quality control, excluding images with severe artifacts or resolution issues</li>
            <li><strong>Tier 2 - Expert Review:</strong> Four independent ophthalmologists independently graded each image, recording the presence or absence of specific pathologies</li>
            <li><strong>Tier 3 - Specialist Verification:</strong> Two senior retinal specialists, each with over 20 years of clinical experience, provided final verification and arbitration of any disagreements</li>
        </ul>
        <p>This comprehensive approach ensures diagnostic accuracy comparable to clinical standards.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <h3>Inter-Grader Reliability</h3>
        <p>To assess the reliability of the grading system, a subset of 993 scans was independently graded by two ophthalmologists. Any disagreements were resolved by a senior retinal specialist. This process demonstrated high inter-grader agreement, validating the consistency of the dataset labels.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Technical Implementation
    st.markdown('<div class="section-header"><h2>Technical Implementation</h2></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>AI Model Architecture</h4>
            <p>The platform utilizes state-of-the-art deep learning models trained on the comprehensive OCT dataset. The models are designed to recognize subtle patterns and features that may be missed by human observers.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>Image Processing</h4>
            <p>Advanced image preprocessing techniques ensure optimal model performance, including normalization, augmentation, and quality enhancement while preserving diagnostic information.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>Performance Metrics</h4>
            <p>The models achieve high accuracy across all disease categories, with robust performance on both common and rare presentations of retinal conditions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>Continuous Improvement</h4>
            <p>The models are regularly updated with new data and improved algorithms to maintain the highest diagnostic accuracy and clinical relevance.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Clinical Applications
    st.markdown('<div class="section-header"><h2>Clinical Applications</h2></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <h3>Primary Care & Screening</h3>
        <p>The platform can be used in primary care settings for initial screening of patients at risk for retinal diseases, enabling early referral to specialists when needed.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <h3>Specialist Practice</h3>
        <p>Ophthalmologists and retinal specialists can use the platform as a second opinion tool, helping to confirm diagnoses and identify subtle changes that may require immediate attention.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <h3>Research & Education</h3>
        <p>The platform serves as an educational tool for medical students and researchers, providing access to a large, well-annotated dataset of retinal pathologies.</p>
    </div>
    """, unsafe_allow_html=True)

# Prediction Page
elif app_mode == "Disease Identification":
    st.markdown("""
    <div class="main-header">
        <h1>Disease Identification</h1>
        <p>Upload your OCT image for TensorFlow-powered deep learning analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Upload Section
    st.markdown('<div class="section-header"><h2>Upload Your OCT Image</h2></div>', unsafe_allow_html=True)
    
    test_image = st.file_uploader(
        "Choose an OCT image file:",
        type=['jpg', 'jpeg', 'png'],
        help="Supported formats: JPG, JPEG, PNG"
    )
    
    # Prediction Section
    if test_image is not None:
        st.markdown('<div class="section-header"><h2>Image Preview</h2></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(test_image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h4>Ready for Analysis</h4>
                <p>Your image has been uploaded successfully. Click the <strong>Analyze Image</strong> button below to get started with the AI-powered diagnosis.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Predict button
        if st.button("Analyze Image", use_container_width=True):
            with st.spinner("AI is analyzing your image..."):
                # Save to a temporary file and get its path
                with tempfile.NamedTemporaryFile(delete=False, suffix=test_image.name) as tmp_file:
                    tmp_file.write(test_image.read())
                    temp_file_path = tmp_file.name
                
                result_index = model_prediction(temp_file_path)
                
                if result_index is not None:
                    # Reading Labels
                    class_name = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
                    disease_name = class_name[result_index]
                    
                    # Success message with styling
                    st.markdown(f"""
                    <div class="stSuccess">
                        Analysis Complete! Result: <strong>{disease_name}</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Disease Information
                    st.markdown('<div class="section-header"><h2>Detailed Analysis</h2></div>', unsafe_allow_html=True)
                    
                    # Recommendation
                    with st.expander(f"Learn More About {disease_name}", expanded=True):
                        # CNV
                        if result_index == 0:
                            st.markdown("""
                            <div class="disease-card">
                                <h4>Choroidal Neovascularization (CNV)</h4>
                                <p>OCT scan showing <em>CNV with subretinal fluid.</em></p>
                            </div>
                            """, unsafe_allow_html=True)
                            st.image(test_image, caption="CNV Analysis", use_column_width=True)
                            st.markdown(cnv)
                        
                        # DME
                        elif result_index == 1:
                            st.markdown("""
                            <div class="disease-card">
                                <h4>Diabetic Macular Edema (DME)</h4>
                                <p>OCT scan showing <em>DME with retinal thickening and intraretinal fluid.</em></p>
                            </div>
                            """, unsafe_allow_html=True)
                            st.image(test_image, caption="DME Analysis", use_column_width=True)
                            st.markdown(dme)
                        
                        # DRUSEN
                        elif result_index == 2:
                            st.markdown("""
                            <div class="disease-card">
                                <h4>Drusen (Early AMD)</h4>
                                <p>OCT scan showing <em>drusen deposits in early AMD.</em></p>
                            </div>
                            """, unsafe_allow_html=True)
                            st.image(test_image, caption="Drusen Analysis", use_column_width=True)
                            st.markdown(drusen)
                            
                        # NORMAL
                        elif result_index == 3:
                            st.markdown("""
                            <div class="disease-card">
                                <h4>Normal Retina</h4>
                                <p>OCT scan showing a <em>normal retina with preserved foveal contour.</em></p>
                            </div>
                            """, unsafe_allow_html=True)
                            st.image(test_image, caption="Normal Retina Analysis", use_column_width=True)
                            st.markdown(normal)
                else:
                    st.error("Error during analysis. Please try again or contact support.")
    
    else:
        st.markdown("""
        <div class="feature-card">
            <h3>Ready to Upload</h3>
            <p>Please upload an OCT image above to begin the analysis process. Our AI model will classify the image and provide detailed insights about any detected retinal conditions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Instructions
        st.markdown('<div class="section-header"><h2>How It Works</h2></div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h4>Upload</h4>
                <p>Select your OCT image file from your device</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h4>Analyze</h4>
                <p>Our AI model processes the image using advanced algorithms</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="feature-card">
                <h4>Results</h4>
                <p>Get detailed analysis and recommendations</p>
            </div>
            """, unsafe_allow_html=True)