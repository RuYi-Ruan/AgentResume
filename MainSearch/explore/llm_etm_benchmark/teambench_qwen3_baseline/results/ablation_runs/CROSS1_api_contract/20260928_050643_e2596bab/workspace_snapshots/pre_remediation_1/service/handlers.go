package main

import (
	"encoding/json"
	"net/http"
	"strconv"
)

func listUsers(w http.ResponseWriter, r *http.Request) {
	items := []User{
		{UserID: 1, Name: "Alice", Email: "alice@example.com"},
		{UserID: 2, Name: "Bob", Email: "bob@example.com"},
	}
	page := 1
	pageStr := r.URL.Query().Get("page")
	if pageStr != "" {
		page, _ = strconv.Atoi(pageStr)
	}

	resp := PaginatedResponse{
		Data:  items,
		Next:  "cursor_" + strconv.Itoa(page+1),
		Total: len(items),
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func createUser(w http.ResponseWriter, r *http.Request) {
	var item User
	if err := json.NewDecoder(r.Body).Decode(&item); err != nil {
		w.WriteHeader(422)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(ErrorResponse{
			Errors: []string{"Invalid JSON body", "Request body required"},
		})
		return
	}
	w.WriteHeader(201)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(item)
}
