// https://en.wikipedia.org/wiki/Single-responsibility_principle
package main

import (
	"fmt"
	"io/ioutil"
	"net/url"
	"strings"
)

// original responsibility
var entryCount = 0

type Journal struct {
	entries []string
}

func (j *Journal) String() string {
	return strings.Join(j.entries, "\n")
}
func (j *Journal) AddEntry(text string) int {
	entryCount++
	entry := fmt.Sprintf("%d: %s",
		entryCount,
		text)
	j.entries = append(j.entries, entry)
	return entryCount
}
func (j *Journal) RemoveEntry(index int) {
	// ...
}

// breaks srp
func (j *Journal) Save(filename string) {
	_ = ioutil.WriteFile(filename,
		[]byte(j.String()), 0644)
}
func (j *Journal) Load(filename string) {
	// ...
}
func (j *Journal) LoadFromWeb(url *url.URL) {
	// ...
}

// srp #1
var lineSeparator = "\n"

func SaveToFile(j *Journal, filename string) {
	_ = ioutil.WriteFile(filename,
		[]byte(strings.Join(j.entries, lineSeparator)), 0644)
}

// srp #2
type Persistence struct {
	lineSeparator string
}

func (p *Persistence) saveToFile(j *Journal, filename string) {
	_ = ioutil.WriteFile(filename,
		[]byte(strings.Join(j.entries, p.lineSeparator)), 0644)
}

func main() {
	j := Journal{}
	j.AddEntry("I cried today.")
	j.AddEntry("I ate a bug")
	fmt.Println(strings.Join(j.entries, "\n"))

	// separate function
	SaveToFile(&j, "journal.txt")

	// separate class
	p := Persistence{"\n"}
	p.saveToFile(&j, "journal.txt")
}
