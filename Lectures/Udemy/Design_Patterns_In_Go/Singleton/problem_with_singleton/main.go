package main

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"sync"
)

// think of a module as a singleton
type Database interface {
	GetPopulation(name string) int
}

type singletonDatabase struct {
	capitals map[string]int
}

func (db *singletonDatabase) GetPopulation(
	name string) int {
	return db.capitals[name]
}

// both init and sync.Once are thread-safe
// but only sync.Once is lazy
var once sync.Once
var instance Database

func readData(path string) (map[string]int, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	scanner.Split(bufio.ScanLines)

	result := map[string]int{}

	for scanner.Scan() {
		k := scanner.Text()
		scanner.Scan()
		v, _ := strconv.Atoi(scanner.Text())
		result[k] = v
	}

	return result, nil
}

func GetSingletonDatabase() Database {
	once.Do(func() {
		db := singletonDatabase{}
		absPath, _ := filepath.Abs("./Lectures/Udemy/Design_Patterns_In_Go/Singleton/capitals.txt")
		fmt.Println(absPath)
		caps, err := readData(absPath)
		if err == nil {
			db.capitals = caps
		} else {
			fmt.Println(err)
		}
		instance = &db
	})
	return instance
}

func GetTotalPopulation(cities []string) int {
	result := 0
	for _, city := range cities {
		// problem-2 : not following DIP (Dependency Inversion Principle) principle.
		// which means high-level module should not depend on low-level module.
		// GetSingletonDatabase() is a low-level module, and GetTotalPopulation() is a high-level module.
		// when testing "GetTotalPopulation", you are not just testing "GetTotalPopulation", but also "GetSingletonDatabase".
		result += GetSingletonDatabase().GetPopulation(city)
	}
	return result
}

func main() {
	cities := []string{"Seoul", "Mexico City"}
	tp := GetTotalPopulation(cities)
	// unit-test: testing on live data
	ok := tp == (17500000 + 17400000) // problem-1 : test depends on live data.
	fmt.Println(ok)

}
