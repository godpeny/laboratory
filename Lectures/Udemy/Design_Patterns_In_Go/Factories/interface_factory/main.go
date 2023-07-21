package main

import "fmt"

type person struct {
	Name string
	Age  int
}

type tiredPerson struct {
	Name string
	Age  int
}

type Person interface {
	SayHello()
}

func NewPersonFactory(name string, age int) Person {
	if age > 100 {
		return &tiredPerson{name, age}
	}
	return &person{name, age}
}

func (p *person) SayHello() {
	fmt.Printf("Hi, my name is %s. I am %d years old\n", p.Name, p.Age)
}

func (tp *tiredPerson) SayHello() {
	fmt.Printf("Hi, my name is %s. I am tired\n", tp.Name)
}

func main() {
	// initialize directly
	p := NewPersonFactory("John", 30)
	fmt.Println(p)

	tp := NewPersonFactory("OldJohn", 101)
	fmt.Println(tp)
}
