//this code must be in the same file as server.py since they are bound to each other

import socket
import sys

server_ip = sys.argv[1]
server_port = int(sys.argv[2])


client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# connect the socket to the server's address and port
server_address = (server_ip, server_port)
client_socket.connect(server_address)


message = input("Enter a message to send to the server: ")
client_socket.sendall(message.encode())


data = client_socket.recv(1024)
print(f"Received response from server: {data.decode()}")


client_socket.close()

command = input("Enter 'quit' to close the client: ")
if command == "quit":
    client_socket.close()
    sys.exit(0)
