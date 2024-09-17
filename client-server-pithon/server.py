import socket
import sys

# create a socket(ipv4 and TCP socket)
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# bind the socket to a specific address and port
server_address = ('', 8080)  
server_socket.bind(server_address)


server_socket.listen(3) 
print("Waiting for connections...")


while True:
    client_socket, client_address = server_socket.accept()
    print(f"Connection from {client_address}")

   
    while True:
        data = client_socket.recv(1024)
        if not data:
            break
        print(f"Received message from {client_address}: {data.decode()}")

        
        response = f"Server received message: {data.decode()}"
        client_socket.sendall(response.encode())

    
    client_socket.close()

