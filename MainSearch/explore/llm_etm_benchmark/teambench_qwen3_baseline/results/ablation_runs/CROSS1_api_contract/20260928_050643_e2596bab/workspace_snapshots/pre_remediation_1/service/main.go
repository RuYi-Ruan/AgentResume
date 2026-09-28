package main

import (
	"fmt"
	"net/http"
	"os"
)

func main() {
	port := "8080"
	if p := os.Getenv("PORT"); p != "" {
		port = p
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/users", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "GET" {
			listUsers(w, r)
		} else if r.Method == "POST" {
			createUser(w, r)
		} else {
			w.WriteHeader(405)
		}
	})

	fmt.Printf("Server starting on port %s\n", port)
	http.ListenAndServe(":"+port, mux)
}
