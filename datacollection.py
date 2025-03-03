import json
import os

# Create a directory for our data
os.makedirs("data", exist_ok=True)

# Sample JioPay dummy FAQ data
dummy_faqs = [
    {
        "source": "https://jiopay.com/faq/account-setup",
        "title": "Account Setup FAQ",
        "content": """
        How do I create a JioPay account?
        To create a JioPay account, download the JioPay app from the App Store or Google Play Store. Open the app and click on 'Sign Up'. Enter your mobile number and complete the verification process. Set up your profile and you're good to go!
        
        What documents do I need to set up a JioPay account?
        To set up a JioPay account, you'll need a valid mobile number, an email address, and a government-issued ID such as Aadhaar card, PAN card, or driver's license for KYC verification.
        
        Is there a fee to create a JioPay account?
        No, creating a JioPay account is completely free. There are no setup fees or monthly maintenance charges for basic accounts.
        """
    },
    {
        "source": "https://jiopay.com/faq/payments",
        "title": "Payments FAQ",
        "content": """
        How do I send money to someone using JioPay?
        To send money, open the JioPay app and tap on 'Send Money'. Enter the recipient's mobile number or select from your contacts. Enter the amount you wish to send and confirm the transaction with your PIN or biometric authentication.
        
        What are the transaction limits on JioPay?
        For basic KYC users, the daily transaction limit is ₹10,000. For fully verified accounts, the daily limit is ₹1,00,000. Monthly limits may apply based on your account type.
        
        Does JioPay charge any transaction fees?
        JioPay does not charge any fees for UPI transactions. However, merchant payments through credit cards may incur a small fee of 1-2% depending on the merchant category.
        """
    },
    {
        "source": "https://jiopay.com/faq/security",
        "title": "Security FAQ",
        "content": """
        How secure is JioPay?
        JioPay uses industry-standard 256-bit encryption to secure all transactions. We also implement multi-factor authentication, biometric verification, and real-time fraud detection systems to ensure your money stays safe.
        
        What should I do if I lose my phone?
        If you lose your phone, immediately contact JioPay customer support at 1800-XXX-XXXX to block your account. You can also log in to JioPay web portal from another device and temporarily disable mobile transactions.
        
        How can I change my JioPay PIN?
        To change your PIN, go to Settings > Security > Change PIN in the JioPay app. You'll need to enter your current PIN, then set and confirm your new PIN.
        """
    },
    {
        "source": "https://jiopay.com/faq/troubleshooting",
        "title": "Troubleshooting FAQ",
        "content": """
        Why is my transaction failing?
        Transactions may fail due to several reasons: insufficient balance, daily limits exceeded, network issues, or bank server downtime. Check your balance, ensure you haven't exceeded daily limits, and try again later if network issues persist.
        
        My payment is stuck 'in process'. What should I do?
        If your payment is stuck, don't retry the transaction immediately. Wait for 24 hours as most pending transactions resolve automatically. You can check the status in Transaction History. If it remains unresolved after 24 hours, contact customer support.
        
        How do I raise a dispute for a failed transaction?
        To raise a dispute, go to Transaction History, select the failed transaction, and tap on 'Raise Dispute'. Provide the necessary details and submit. Our team will resolve most disputes within 3-5 business days.
        """
    },
    {
        "source": "https://jiopay.com/business/merchant-services",
        "title": "Merchant Services",
        "content": """
        How can I register as a JioPay merchant?
        To register as a JioPay merchant, visit our Business Portal at jiopay.com/business and click on 'Become a Merchant'. You'll need to provide your business details, bank account information, and complete the verification process. Our team will set up your merchant account within 2-3 business days.
        
        What are the fees for JioPay merchant services?
        JioPay offers competitive rates for merchant services. UPI transactions are completely free. For card transactions, we charge 1-2% based on your business category. High-volume merchants may qualify for special rates. Contact our sales team for a customized quote.
        
        How do I get a JioPay QR code for my business?
        Once your merchant account is approved, you can download and print your unique JioPay QR code from the Merchant Portal. We also offer physical QR standees and integration options for your website or app. For premium merchants, we provide free QR standees and promotional materials.
        """
    }
]

# Save dummy data to a JSON file
with open("data/dummy_jiopay_faqs.json", "w") as f:
    json.dump(dummy_faqs, f, indent=4)

print("Dummy JioPay data created and saved to data/dummy_jiopay_faqs.json")