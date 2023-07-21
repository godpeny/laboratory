package main

import "fmt"

type Person struct {
	Name     string
	Age      int
	EyeCount int // default value
}

func NewPersonFactory(name string, age int) *Person {
	if age < 16 {
		// validate age if needed
	}
	return &Person{name, age, 2}
}

func main() {
	// initialize directly
	p := NewPersonFactory("John", 30)
	fmt.Println(p)

	// use a constructor
	p2 := NewPersonFactory("Jane", 21)
	p2.Age = 30
	fmt.Println(p2)
}
