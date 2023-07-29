package main

import "fmt"

type Image interface {
	Draw()
}

type Bitmap struct {
	filename string
}

func (b *Bitmap) Draw() {
	fmt.Println("Drawing image", b.filename)
}

func NewBitmap(filename string) *Bitmap {
	fmt.Println("Loading image from", filename)
	return &Bitmap{filename: filename}
}

func DrawImage(image Image) {
	fmt.Println("About to draw the image")
	image.Draw()
	fmt.Println("Done drawing the image")
}

// LazyBitmap is Virtual Proxy
// why virtual proxy? because when you create a lazy bitmap using the new lazy bitmap function,
// it("bitmap") hasn't been materialized yet.
// it's only constructed whenever somebody explicitly asks for it and in this case, calling the "Draw" function.
type LazyBitmap struct {
	filename string
	bitmap   *Bitmap
}

func (l *LazyBitmap) Draw() {
	if l.bitmap == nil {
		l.bitmap = NewBitmap(l.filename)
	}
	l.bitmap.Draw()
}

func NewLazyBitmap(filename string) *LazyBitmap {
	return &LazyBitmap{filename: filename}
}

func main() {
	//bmp := NewBitmap("demo.png")
	bmp := NewLazyBitmap("demo.png")
	DrawImage(bmp)
	DrawImage(bmp) // this time, it's not going to load the image again.
}
