import threading
import requests
import os

class DownloadThread(threading.Thread):
    def __init__(self, url, output_file):
        super(DownloadThread, self).__init__()
        self.url = url
        self.output_file = output_file

    def run(self):
        try:
            # Download the file content
            response = requests.get(self.url)
            content = response.text

            # Get the current working directory and construct the full path
            current_directory = os.path.dirname(os.path.abspath(__file__))
            full_path = os.path.join(current_directory, self.output_file)

            # Save the content to the specified file
            with open(full_path, 'w') as file:
                file.write(content)

            print(f"Downloaded {self.url}. Saved to {full_path}")

        except Exception as e:
            print(f"Error downloading {self.url}: {e}")

class DecryptThread(threading.Thread):
    def __init__(self, input_file, decrypted_data):
        super(DecryptThread, self).__init__()
        self.input_file = input_file
        self.decrypted_data = decrypted_data

    def run(self):
        try:
            # Get the current working directory and construct the full path
            current_directory = os.path.dirname(os.path.abspath(__file__))
            full_path = os.path.join(current_directory, self.input_file)

            # Open the file and read its content
            with open(full_path, 'r') as file:
                encrypted_content = file.read()

            # Decrypt the content using Caesar cipher with offset 8
            decrypted_content = self.caesar_cipher_decrypt(encrypted_content, shift=8)

            # Store the input file name and decrypted content in the shared data structure
            self.decrypted_data.append((self.input_file, decrypted_content))

            print(f"Decrypted {self.input_file} and stored in memory")

        except Exception as e:
            print(f"Error decrypting {self.input_file}: {e}")

    def caesar_cipher_decrypt(self, text, shift=8):
        result = ''
        for char in text:
            if char.isalpha():
                ascii_offset = ord('a') if char.islower() else ord('A')
                result += chr((ord(char) - ascii_offset - shift) % 26 + ascii_offset)
            else:
                result += char
        return result

class Combiner(threading.Thread):
    def __init__(self, decrypted_data, output_file):
        super(Combiner, self).__init__()
        self.decrypted_data = decrypted_data
        self.output_file = output_file

    def run(self):
        try:
            # Sort the decrypted data in the correct order based on the input file names
            sorted_data = sorted(self.decrypted_data, key=lambda x: x[0])

            # Combine the decrypted data in the correct order
            combined_content = "\n".join(content for _, content in sorted_data)

            # Get the current working directory and construct the full path
            current_directory = os.path.dirname(os.path.abspath(__file__))
            full_path = os.path.join(current_directory, self.output_file)

            # Write the combined content to the output file
            with open(full_path, 'w') as file:
                file.write(combined_content)

            print(f"Combined and saved the content to {full_path}")

        except Exception as e:
            print(f"Error combining and saving content: {e}")

# URLs and corresponding output files
urls = [
    "https://advancedpython.000webhostapp.com/s1.txt",
    "https://advancedpython.000webhostapp.com/s2.txt",
    "https://advancedpython.000webhostapp.com/s3.txt"
]
output_files = ["s1_enc.txt", "s2_enc.txt", "s3_enc.txt"]

# Specify the directory to save the files
specified_directory = r"D:\projectsvisualstudio\pithon project 3\problem 1"

# Create and start DownloadThread instances
download_threads = []
for url, output_file in zip(urls, output_files):
    download_thread = DownloadThread(url, output_file)
    download_threads.append(download_thread)

# Wait for all download threads to finish
for download_thread in download_threads:
    download_thread.start()
for download_thread in download_threads:
    download_thread.join()

# Create a list to store decrypted data in memory
decrypted_data = []

# Create and start DecryptThread instances
decrypt_threads = []
for output_file in output_files:
    decrypt_thread = DecryptThread(output_file, decrypted_data)
    decrypt_threads.append(decrypt_thread)

# Wait for all decrypt threads to finish
for decrypt_thread in decrypt_threads:
    decrypt_thread.start()
for decrypt_thread in decrypt_threads:
    decrypt_thread.join()

# File to write the combined content
output_file_combined = "s_final.txt"

# Create and start Combiner instance
combiner = Combiner(decrypted_data, output_file_combined)
combiner.start()

# Wait for the Combiner thread to finish
combiner.join()

# Display a message indicating the process is complete
print("Combining and saving process completed.")
