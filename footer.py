import streamlit as st

def render_footer():
    """Render a consistent footer across all pages."""
    st.markdown("""
        <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background: linear-gradient(90deg, #0f2027, #203a43, #2c5364);
            color: #ffffff;
            text-align: center;
            padding: 10px 0 8px 0;
            font-size: 14px;
            letter-spacing: 0.5px;
            z-index: 9999;
            border-top: 1px solid #4a9aba;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.3);
        }
        .footer span {
            color: #a8d8ea;
            font-weight: 600;
        }
        /* Push page content up so footer doesn't overlap it */
        .main .block-container {
            padding-bottom: 60px;
        }
        </style>

        <div class="footer">
            © 2026 <span>Rahul Ram</span> &nbsp;|&nbsp; AI Credit Card Fraud Detection System
            &nbsp;|&nbsp; All Rights Reserved
        </div>
    """, unsafe_allow_html=True)
