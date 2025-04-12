from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
import os
import ast

class ToolDetectionPipeline:
    def __init__(self, image_urls: list, description: str = ""):
        self.image_urls = image_urls
        self.description = description

        # Load the LLM (OpenRouter GPT-4o-mini)
        self.llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENAI_API_KEY"),
            model="openai/gpt-4o-mini"
        )

        # Agent role and backstory embedded in prompt
        self.prompt = ChatPromptTemplate.from_template("""
            You are {role}. Your mission: {goal}

            Backstory:
            {backstory}

            Task:
            Analyze the following screenshot images to identify which of these tools are being used:
            - Google Sheets
            - Gmail
            - Freshdesk

            Image URLs:
            {image_urls}

            Instructions:
            - Examine each image carefully for UI elements, layouts, and visual cues specific to each tool
            - Output ONLY a comma-separated list of the tools detected across ALL images
            - Do NOT repeat tools in the output even if they appear in multiple images
            - Do NOT include any explanations, headers, or additional text
            - If no tools are detected, output an empty string

            Example output: "Google Sheets, Gmail"

            Begin output:"""
        )

        self.agent_profile = {
            "role": "Software Tool Identification Specialist",
            "goal": "Accurately identify which productivity tools (Google Sheets, Gmail, Freshdesk) are present in the provided screenshots",
            "backstory": "You are an expert UI/UX analyst who specializes in identifying software tools from screenshots. You have extensive experience recognizing the distinctive interfaces of Google Workspace applications and customer service platforms like Freshdesk.",
        }

        # Combine prompt and model
        self.chain: Runnable = self.prompt | self.llm

    def run(self):
        formatted_goal = self.agent_profile["goal"]
        formatted_backstory = self.agent_profile["backstory"]
        formatted_image_urls = "\n".join([f"- {url}" for url in self.image_urls])

        input_data = {
            "image_urls": formatted_image_urls,
            "description": self.description,
            "role": self.agent_profile["role"],
            "goal": formatted_goal,
            "backstory": formatted_backstory
        }

        result = self.chain.invoke(input_data)
        print("Raw LLM output:\n", result.content)

        return self.parse_tools(result.content)

    def parse_tools(self, content: str):
        try:
            # Clean and normalize the response
            cleaned_content = content.strip()
            
            # Remove any markdown code block formatting if present
            if cleaned_content.startswith("```") and cleaned_content.endswith("```"):
                cleaned_content = cleaned_content[3:-3].strip()
                
            # Further cleanup if needed
            return cleaned_content
        except Exception as e:
            print(f"Error parsing tools: {e}")
            return ""

# Main function
def detect_tools(image_urls: list, description: str = ""):
    pipeline = ToolDetectionPipeline(image_urls, description)
    return pipeline.run()
