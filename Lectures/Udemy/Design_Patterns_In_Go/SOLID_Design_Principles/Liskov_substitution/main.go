/*
 https://en.wikipedia.org/wiki/Liskov_substitution_principle
 https://stackify.com/solid-design-liskov-substitution-principle

 The principle defines that objects of a superclass shall be replaceable with objects of its subclasses without breaking the application.
 That requires the objects of your subclasses to behave in the same way as the objects of your superclass.

 자식 객체는 부모 객체의 역할을 포함해야 한다.
 자식 클래스가 항상 부모 클래스의 역할을 충실히 수행해야 한다.
*/

package main

import "fmt"

type Sized interface {
	GetWidth() int
	SetWidth(width int)
	GetHeight() int
	SetHeight(height int)
}

type Rectangle struct {
	width, height int
}

// vvv !! POINTER
func (r *Rectangle) GetWidth() int {
	return r.width
}

func (r *Rectangle) SetWidth(width int) {
	r.width = width
}

func (r *Rectangle) GetHeight() int {
	return r.height
}

func (r *Rectangle) SetHeight(height int) {
	r.height = height
}

// modified LSP
// If a function takes an interface and
// works with a type T that implements this
// interface, any structure that aggregates T
// should also be usable in that function.
type Square struct {
	Rectangle
}

func NewSquare(size int) *Square {
	sq := Square{}
	sq.width = size
	sq.height = size
	return &sq
}

func (s *Square) SetWidth(width int) {
	s.width = width
	s.height = width
}

func (s *Square) SetHeight(height int) {
	s.width = height
	s.height = height
}

type Square2 struct {
	size int
}

func (s *Square2) Rectangle() Rectangle {
	return Rectangle{s.size, s.size}
}

func UseIt(sized Sized) {
	width := sized.GetWidth()
	sized.SetHeight(10)
	expectedArea := 10 * width
	actualArea := sized.GetWidth() * sized.GetHeight()
	fmt.Print("Expected an area of ", expectedArea,
		", but got ", actualArea, "\n")
}

func main() {
	rc := &Rectangle{2, 3}
	UseIt(rc)

	sq := NewSquare(5)
	UseIt(sq)
}
