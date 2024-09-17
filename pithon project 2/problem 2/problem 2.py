from itertools import permutations

def read_word():
    return input("Enter a word: ").strip()

def load_dictionary(file_path='dictionary.txt'):
    try:
        with open(file_path, 'r') as file:
            return set(word.strip().lower() for word in file)
    except FileNotFoundError:
        print(f"Error: Could not find the dictionary file '{file_path}'.")
        return set()

def create_anagrams(word):
    return set(''.join(anagram) for anagram in permutations(word))

def filter_meaningful_words(anagrams, dictionary):
    return filter(lambda word: word in dictionary, anagrams)

def display_meaningful_words(meaningful_words):
    if meaningful_words:
        print("Meaningful words:")
        for word in meaningful_words:
            print(word)
    else:
        print("No meaningful words found.")

def main():
    word = read_word().lower()
    dictionary_path = 'D:\projectsvisualstudio\pithon project 2\problem 2\dictionary.txt'  # Replace with the actual path to your dictionary file
    dictionary = load_dictionary(dictionary_path)
    anagrams = create_anagrams(word)
    meaningful_words = filter_meaningful_words(anagrams, dictionary)
    display_meaningful_words(meaningful_words)

if __name__ == "__main__":
    main()
