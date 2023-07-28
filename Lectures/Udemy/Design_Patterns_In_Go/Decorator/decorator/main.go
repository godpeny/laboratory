package main

import "fmt"

type Shape interface {
	Render() string
}

type Circle struct {
	Radius float32
}

func (c *Circle) Render() string {
	return fmt.Sprintf("Circle of radius %f",
		c.Radius)
}

func (c *Circle) Resize(factor float32) {
	c.Radius *= factor
}

type Square struct {
	Side float32
}

func (s *Square) Render() string {
	return fmt.Sprintf("Square with side %f", s.Side)
}

// possible, but not generic enough
type ColoredSquare struct {
	Square
	Color string
}

// real decorator
type ColoredShape struct {
	Shape Shape
	Color string
}

func (c *ColoredShape) Render() string {
	return fmt.Sprintf("%s has the color %s",
		c.Shape.Render(), c.Color)
}

// real decorator 2
type TransparentShape struct {
	Shape        Shape
	Transparency float32
}

func (t *TransparentShape) Render() string {
	return fmt.Sprintf("%s has %f%% transparency",
		t.Shape.Render(), t.Transparency*100.0)
}

func main() {
	circle := Circle{2}
	fmt.Println(circle.Render())
	circle.Resize(2.5) // possible

	redCircle := ColoredShape{&circle, "Red"}
	fmt.Println(redCircle.Render())
	// redCircle.Resize(2.5)  not possible (downside of decorator pattern)
	// redCircle.Shape.Resize(2)  not possible (downside of decorator pattern)

	rhsCircle := TransparentShape{&redCircle, 0.5} // decorator over decorator
	fmt.Println(rhsCircle.Render())
}
