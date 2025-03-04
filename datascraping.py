from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import json

# Configure Chrome options
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
# options.add_experimental_argument("excludeSwitches", ["enable-automation"])

# Initialize driver (update chromedriver path)
driver = webdriver.Chrome()
driver.get("https://www.jiopay.in/business/help-center")

# Wait for page to load
time.sleep(5)

# Find all question elements
questions = WebDriverWait(driver, 20).until(
    EC.presence_of_all_elements_located((By.XPATH, "//div[contains(@class, 'r-1m36w87') and contains(@class, 'r-13yce4e')]"))
)

faq_data = []

# Process each question
for index, question in enumerate(questions):
    try:
        # Scroll to question
        driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", question)
        time.sleep(0.5)
        
        # Click to expand answer
        question.click()
        time.sleep(0.5)  # Allow animation
        
        # Find answer element
        answer = WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.XPATH, f"//div[contains(@class, 'r-14lw9ot') and contains(@class, 'r-1m36w87')][{index+1}]/following-sibling::div[contains(@class, 'r-gy4na3')][1]"))
        )
        
        # Extract text
        faq_data.append({
            "question": question.text,
            "answer": answer.text.strip()
        })
        
    except Exception as e:
        print(f"Error processing question {index+1}: {str(e)}")
        continue

# Convert faq_data to JSON format with multiple FAQs
json_data = []
for i in range(0, len(faq_data), 2): # Assuming you want to group FAQs in pairs
    content = "\n\n".join([f"{item['question']}\n{item['answer']}" for item in faq_data[i:i+2]])
    json_data.append({
        "source": "https://www.jiopay.in/business/help-center",
        "title": f"JioPay Business FAQ {i//2+1}",
        "content": content
    })

# Save JSON to file
with open('jiopay_faq.json', 'w', encoding='utf-8') as jsonfile:
    json.dump(json_data, jsonfile, indent=4)

driver.quit()
print("Scraping completed. Data saved to jiopay_faq.json")
