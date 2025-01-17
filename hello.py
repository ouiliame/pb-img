# File paths and constants
WORKFLOW_FILE = "pb.json"
DEFAULT_PROMPT = """
A realistic photograph of a young man mid-air, leaping onto a neatly made bed with his body oriented naturally—head aimed
toward the pillows and feet toward the foot of the bed. His arms are extended forward as if preparing to land, and his body
is slightly angled downward to emphasize forward motion. The bed features soft, slightly rumpled linens, with pillows at the
headboard clearly visible. The room is bright and minimalist, with sunlight streaming through a large window and simple decor
like bedside tables, books, and plants to create a cozy, relatable atmosphere. The man’s expression should convey playful joy,
enhancing the dynamic and realistic scene.
"""
REPLICATE_MODEL = "fofr/any-comfyui-workflow-a100:d7cbe5383efd0b00d1a147b96cc0eabdc479f67359bf594d98e3bbc3df52233f"
GPT4_MODEL = "gpt-4o"
MAX_TOKENS = 300
OUTPUT_QUALITY = 95

import json
import replicate
import base64
from openai import OpenAI
from PIL import Image
import io

def load_workflow(workflow_file):
    """Load workflow from JSON file"""
    with open(workflow_file) as f:
        return json.load(f)

def save_workflow(workflow_file, workflow):
    """Save workflow to JSON file"""
    with open(workflow_file, 'w') as f:
        json.dump(workflow, f, indent=2)

def update_prompt(workflow, new_prompt):
    """Update the prompt in the workflow"""
    workflow["6"]["inputs"]["text"] = new_prompt
    return workflow

def generate_image(workflow):
    """Generate single image using ComfyUI workflow"""
    return replicate.run(
        REPLICATE_MODEL,
        input={
            "workflow_json": json.dumps(workflow),
            "randomise_seeds": True,
            "return_temp_files": False,
            "output_quality": OUTPUT_QUALITY
        }
    )

def analyze_with_gpt4(image_path, prompt):
    """Get GPT-4 feedback on image"""
    client = OpenAI()
    
    # Read and convert image to base64
    with open(image_path, 'rb') as img_file:
        img_bytes = img_file.read()
        base64_image = base64.b64encode(img_bytes).decode('utf-8')
    
    # Create message with base64 image in proper format
    message_content = [
        {
            "type": "text",
            "text": f"Does this image match this prompt? Suggest improvements: {prompt}"
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        }
    ]
    
    response = client.chat.completions.create(
        model=GPT4_MODEL,
        messages=[{
            "role": "user",
            "content": message_content
        }],
        max_tokens=MAX_TOKENS
    )
    return response.choices[0].message.content

def save_image(image_data, path):
    """Save image data to file"""
    with open(path, 'wb') as f:
        f.write(image_data.read())

def main():
    # Load and update workflow
    workflow = load_workflow(WORKFLOW_FILE)
    workflow = update_prompt(workflow, DEFAULT_PROMPT)
    save_workflow(WORKFLOW_FILE, workflow)
    
    # Generate and analyze image
    outputs = generate_image(workflow)
    print("Generated images")

    for i, image_data in enumerate(outputs):
        image_path = f'output_{i}.png'
        save_image(image_data, image_path)
        
        feedback = analyze_with_gpt4(image_path, DEFAULT_PROMPT)
        print(f"\nFeedback for image {i}:")
        print(feedback)

if __name__ == "__main__":
    main()
