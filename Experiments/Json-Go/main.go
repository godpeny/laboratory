package main

import (
	"encoding/json"
	"fmt"
)

type Task struct {
	UUID       string  `json:"uuid"`
	Status     string  `json:"status"`
	Task       string  `json:"task"`
	ExecutedBy string  `json:"executedBy"`
	JobID      *string `json:"jobID,omitempty"`
}

type Edge struct {
	UUID          string `json:"uuid"`
	Edge          string `json:"edge"`
	Condition     string `json:"condition"`
	Status        string `json:"status" gorm:"size:50"`
	DesiredStatus string `json:"desiredStatusFromTask" gorm:"size:50"`
	From          string `json:"from" gorm:"size:50"`
	To            string `json:"to" gorm:"size:50"`
}

type Artifact struct {
	UUID         string `json:"uuid"`
	Pipeline     string `json:"pipeline" gorm:"size:50"`
	Task         string `json:"task" gorm:"size:50"`
	ArtifactType string `json:"artifactType" gorm:"size:50"`
	Key          string `json:"key" gorm:"type:longtext"`
	Value        string `json:"value" gorm:"type:longtext"`
}

type Pipeline struct {
	UUID      string     `json:"uuid" `
	Trigger   string     `json:"trigger"`
	Status    string     `json:"status" `
	Edges     []Edge     `json:"edges"`
	Tasks     []Task     `json:"tasks"`
	Artifacts []Artifact `json:"artifacts"`
}

func main() {
	eg := `{
	"uuid": "this_pipeline",
	"trigger": "trigger1",
	"status": "status",
	"edges": [
		{
			"uuid": "e1",
			"edge": "edge",
			"condition": "condition",
			"status": "waiting",
			"desiredStatus": "success",
			"from": "t1",
			"to": "t2"
		}
	],
	"tasks": [
		{
			"uuid": "t1",
			"status": "status",
			"task": "task",
			"executedBy": "XXYYZZ",	
			"jobID": "jobID"
		},
		{
			"uuid": "t2",
			"status": "status",
			"task": "task",
			"executedBy": "XXYYZZ",
			"jobID": "jobID"
		}
	],
	"artifacts": [
		{
			"uuid": "a1",
			"artifactType": "artifactType",
			"pipeline": "this_pipeline",
			"task": "t1",
			"key": "k",
			"value": "v"
		}
	]
}`
	// json to go
	pipe := &Pipeline{}
	err := json.Unmarshal([]byte(eg), pipe)
	if err != nil {
		fmt.Println(err.Error())
		//json: Unmarshal(non-pointer main.Request)
	}

	fmt.Println(pipe)

	// go to json
	p, err := json.Marshal(pipe)
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Println(string(p))

	pipe2 := &Pipeline{}
	err = json.Unmarshal(p, pipe2)
	if err != nil {
		fmt.Println(err)
	}

	fmt.Println(pipe2)
}
