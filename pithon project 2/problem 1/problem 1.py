import os

def decode_content(content):
    replacements = {'!': 's', '@': 'h', '#': 'e', '$': 'r', '%': 'l', '^': 'o', '&': 'c', '*': 'k'}
    decoded_content = ''.join(replacements.get(char, char) for char in content)
    return decoded_content

def filter_words_starting_with_a(words):
    return list(filter(lambda word: word.startswith('a'), words))

def read_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        return None

def main():
    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Combine the script directory with the file name
    file_path = os.path.join(script_dir, 'sherlock.txt')

    print(f"Attempting to read 'sherlock.txt' from the directory: {script_dir}")

    # Step 1: Read content from the file
    content = read_file(file_path)

    if content is not None:
        # Step 2: Decode the content
        decoded_content = decode_content(content)

        # Step 3: Split the content into words
        words = decoded_content.split()

        # Step 4: Filter words starting with 'a'
        filtered_words = filter_words_starting_with_a(words)

        # Step 5: Display the list of filtered words
        print(filtered_words)
    else:
        print(f"Error: 'sherlock.txt' not found in the directory '{script_dir}'.")

if __name__ == "__main__":
    main()
