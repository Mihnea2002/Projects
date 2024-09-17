from functools import reduce

def add_movie(movies, title, rank):
    return movies + [{'title': title, 'rank': rank}]

def display_movies(movies):
    display = lambda movie: print(f"Title: {movie['title']}, Rank: {movie['rank']}")
    list(map(display, movies))

def search_movie(movies, title):
    return list(filter(lambda x: x['title'] == title, movies))

def remove_movie(movies, title):
    return list(filter(lambda x: x['title'] != title, movies))

def main():
    movies = []

    def process_choice(choice):
        nonlocal movies
        if choice == '1':
            title = input("Enter the movie title: ")
            rank = float(input("Enter the movie rank: "))
            movies = add_movie(movies, title, rank)
        elif choice == '2':
            print("\nAll Movies:")
            display_movies(movies)
        elif choice == '3':
            search_title = input("Enter the movie title to search: ")
            search_result = search_movie(movies, search_title)
            print("\nSearch Result:")
            display_movies(search_result)
        elif choice == '4':
            remove_title = input("Enter the movie title to remove: ")
            movies = remove_movie(movies, remove_title)
            print("Movie removed successfully.")
        elif choice == '5':
            print("Exiting the program.")
            return True
        else:
            print("Invalid choice. Please enter a valid option.")
        return False

    while True:
        print("\nOptions:")
        print("1. Add a movie")
        print("2. Display all movies")
        print("3. Search for a movie")
        print("4. Remove a movie")
        print("5. Exit")

        choice = input("Enter your choice: ")
        exit_program = process_choice(choice)
        if exit_program:
            break

if __name__ == "__main__":
    main()
