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

// init() — we could, but it's not lazy

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

// GetTotalPopulation DIP violated
func GetTotalPopulation(cities []string) int {
	result := 0
	for _, city := range cities {
		result += GetSingletonDatabase().GetPopulation(city)
	}
	return result
}

// GetTotalPopulationEx DIP not violated, solve problem from "GetTotalPopulation"
func GetTotalPopulationEx(db Database, cities []string) int {
	result := 0
	for _, city := range cities {
		result += db.GetPopulation(city)
	}
	return result
}

type DummyDatabase struct {
	dummyData map[string]int
}

func (d *DummyDatabase) GetPopulation(name string) int {
	if len(d.dummyData) == 0 {
		d.dummyData = map[string]int{
			"alpha": 1,
			"beta":  2,
			"gamma": 3}
	}
	return d.dummyData[name]
}

func main() {
	names := []string{"alpha", "gamma"} // expect 4
	// solve problem from "problem_with_singleton"
	tp := GetTotalPopulationEx(&DummyDatabase{}, names)
	ok := tp == 4
	fmt.Println(ok)
}
