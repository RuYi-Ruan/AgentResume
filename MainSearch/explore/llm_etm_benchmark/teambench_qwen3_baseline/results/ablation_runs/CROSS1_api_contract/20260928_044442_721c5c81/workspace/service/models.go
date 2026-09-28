package main

type User struct {
	UserID int    `json:"userId"`
    Name  string `json:"name"`
    Email string `json:"email"`
}

type PaginatedResponse struct {
	Data  interface{} `json:"data"`
	Next  string       `json:"next"`
	Total int          `json:"total"`
}

type ErrorResponse struct {
	Errors []string `json:"errors"`
}
