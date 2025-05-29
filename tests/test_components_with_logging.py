# --- STEP 1: Add OpenAI Library Logger Configuration ---
import logging
import sys # To output to stdout

# Get the logger used by the openai library
# Note: For openai versions >= 1.0, the logger name might be more specific
# like "openai.api_requestor" or "openai.http_client", but "openai" should
# often catch general API activity or be a parent logger.
# If you don't see logs, try "httpx" as openai v1 uses httpx for requests.
openai_logger = logging.getLogger("openai") # Try this first
# As a fallback or for more detail, also try:
# httpx_logger = logging.getLogger("httpx")
# httpx_logger.setLevel(logging.DEBUG)


# Set the desired logging level. DEBUG will show detailed request/response info.
openai_logger.setLevel(logging.DEBUG)

# Create a handler to output logs to the console (stdout)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG) # Ensure handler also has DEBUG level

# Create a formatter to make the logs readable
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Add the handler to the logger(s)
# Check if handlers already exist to avoid duplicates if script is run multiple times in an interactive session
if not openai_logger.handlers:
    openai_logger.addHandler(handler)
# if not httpx_logger.handlers: # If you added httpx_logger
#     httpx_logger.addHandler(handler) # Use the same handler or a new one

print("---- OpenAI API call logging is ENABLED (DEBUG level) ----")
# --- End of Logger Configuration ---


# --- STEP 2: Your existing imports and code ---
import numpy as np
from lida.components import Manager # Assuming Manager is here
from llmx import llm, TextGenerationConfig
import os
import json # Moved import json here as it's used later

# --- Global Setup ---
# It's good practice to check for the API key early
if not os.getenv("OPENAI_API_KEY"):
    print("CRITICAL ERROR: The OPENAI_API_KEY environment variable is not set.")
    print("Please set it before running the script. You can do this in your .env file for VS Code or your system environment variables.")
    exit() # Exit if the key isn't found

try:
    print("Initializing LIDA Manager with OpenAI...")
    # Make sure to use a model you have access to and that works well
    # For testing logging, gpt-3.5-turbo or gpt-4o-mini are good choices
    lida = Manager(text_gen=llm("openai", model="gpt-4o-mini")) # Example: using gpt-4o-mini
    print("LIDA Manager initialized successfully.")
except Exception as e:
    print(f"Error initializing LIDA Manager: {e}")
    print("Please ensure your OpenAI API key is valid, has credits, and you have an internet connection.")
    exit()


cars_data_url = "https://raw.githubusercontent.com/uwdata/draco/master/data/cars.csv"


def test_summarizer():
    print("\n--- Running test_summarizer ---")
    textgen_config = TextGenerationConfig(
        model="gpt-4o-mini", # Consistent model for summarization
        n=1, temperature=0, use_cache=False, max_tokens=None)
    print("Summarizing with method 'default'...")
    summary_no_enrich = lida.summarize(
        cars_data_url,
        textgen_config=textgen_config,
        summary_method="default")

    print("Summarizing with method 'llm'...")
    summary_enrich = lida.summarize(cars_data_url,
                                    textgen_config=textgen_config, summary_method="llm")

    print("\nDEBUG: Content of 'summary_enrich' received from LLM:")
    try:
        print(json.dumps(summary_enrich, indent=4))
    except TypeError:
        print(summary_enrich)
    print("---------------------------------------------------\n")

    assert summary_no_enrich != summary_enrich, "Default and LLM summaries should be different"
    assert "dataset_description" in summary_enrich and len(
        summary_enrich["dataset_description"]) > 0, "Enriched summary is missing 'dataset_description' or it's empty"
    print("test_summarizer PASSED")


def test_goals():
    print("\n--- Running test_goals ---")
    # For goals, the model will be inherited from the Manager (gpt-4o-mini in this setup)
    # unless overridden in TextGenerationConfig
    textgen_config = TextGenerationConfig(
        n=1, temperature=0.1, use_cache=False, max_tokens=None)
    print("Summarizing data for goal generation...")
    summary = lida.summarize(
        cars_data_url,
        textgen_config=textgen_config, summary_method="default") # Default summary doesn't hit LLM

    print("Generating goals...")
    goals = lida.goals(summary, n=2, textgen_config=textgen_config) # This will use LLM
    assert len(goals) == 2, f"Expected 2 goals, but got {len(goals)}"
    assert goals[0].question and len(goals[0].question) > 0, "The first goal's question is empty"
    print("test_goals PASSED")
    return summary


def test_vizgen(summary_from_goals_test):
    print("\n--- Running test_vizgen ---")
    textgen_config = TextGenerationConfig(
        model="gpt-4o-mini", # Consistent model for vizgen
        n=1,
        temperature=0.1,
        use_cache=True,
        max_tokens=1000
    )

    print("Generating goals for visualization (using provided summary)...")
    # The goals call here will also use the 'model' from this textgen_config
    goals = lida.goals(summary_from_goals_test, n=2, textgen_config=textgen_config)
    if not goals:
        print("No goals generated for visualization. Skipping vizgen test.")
        return

    print(f"Attempting to visualize for goal: {goals[0].question}")
    charts = lida.visualize(
        summary=summary_from_goals_test,
        goal=goals[0],
        textgen_config=textgen_config, # This uses the model specified above
        library="seaborn")

    assert len(charts) > 0, "No charts were generated"
    first_chart = charts[0]
    print(f"Generated chart code (Seaborn):\n{first_chart.code}")

    assert first_chart.status is True, f"Chart generation status is False. Error: {first_chart.error}"
    assert first_chart.error is None, f"Chart generation reported an error: {first_chart.error}"

    if first_chart.raster:
        assert len(first_chart.raster) > 0, "Chart raster data is empty, though status was True."
        print("Chart raster data found.")
    else:
        print("Warning: Chart raster data was not populated by LIDA. Savefig will attempt to render.")

    temp_file_path = "temp_image.png"
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)

    print(f"Attempting to save chart to {temp_file_path}...")
    try:
        first_chart.savefig(temp_file_path)
        assert os.path.exists(temp_file_path), f"{temp_file_path} was not created by savefig"
        print(f"Chart saved successfully to {temp_file_path}")
        os.remove(temp_file_path)
        print(f"{temp_file_path} cleaned up.")
    except Exception as e:
        print(f"ERROR during savefig: {e}")
        print("This might indicate issues with the generated code or rendering backend (e.g., matplotlib, seaborn).")
        assert False, f"savefig failed: {e}"

    print("test_vizgen PASSED")


if __name__ == "__main__":
    print("Starting LIDA tests...")
    try:
        test_summarizer()
        summary_for_viz = test_goals()
        if summary_for_viz:
             test_vizgen(summary_for_viz)
        else:
            print("Skipping vizgen test as goal generation might have failed or not returned summary.")
        print("\n========= ALL APPLICABLE TESTS PASSED SUCCESSFULLY! =========")
    except AssertionError as e:
        print(f"\n!!!!!!!!!! TEST FAILED: {e} !!!!!!!!!!!")
    except Exception as e:
        print(f"\n!!!!!!!!!! AN UNEXPECTED ERROR OCCURRED: {e} !!!!!!!!!!!")
        import traceback
        traceback.print_exc()