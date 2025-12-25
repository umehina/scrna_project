//Name: Yinan Zhu
//Date added: 12/06/2025 and 12/07/2025
// Disclaimer: we consulted ChatGPT for many parser functions in this file.

// distance_matrix_test.go
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strings"
	"testing"

	"gonum.org/v1/gonum/mat"
)

// ---------- generic parser for # / @ format ----------

// ioCase2 holds one test case parsed from the .txt file:
//
//	# Name
//	{"input": ...}
//	@ comment
//	{"output": ...}
type ioCase2 struct {
	Name       string
	InputLine  string
	OutputLine string
}

// loadIOCasesFromFile parses a text file in the "# / @ / JSON" format
// and returns all cases.
func loadIOCasesFromFile(path string) ([]ioCase2, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var (
		cases            []ioCase2
		currentName      string
		pendingInput     string
		waitingForInput  bool
		waitingForOutput bool
	)

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		switch {
		case strings.HasPrefix(line, "#"):
			// Start of a new case
			currentName = strings.TrimSpace(strings.TrimPrefix(line, "#"))
			pendingInput = ""
			waitingForInput = true
			waitingForOutput = false

		case strings.HasPrefix(line, "@"):
			// Comment line between input and output
			// After this, we expect the output JSON line.
			waitingForOutput = true

		default:
			if waitingForInput {
				// This is the input JSON line
				pendingInput = line
				waitingForInput = false
			} else if waitingForOutput {
				// This is the output JSON line → finalize a case
				cases = append(cases, ioCase2{
					Name:       currentName,
					InputLine:  pendingInput,
					OutputLine: line,
				})
				waitingForOutput = false
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return cases, nil
}

// ---------- DistanceMatrix tests ----------

// distanceMatrixInput is the JSON shape we expect in the input line.
// Example:
//
//	{"data":[[1,2],[3,4]]}
type distanceMatrixInput struct {
	Data [][]float64 `json:"data"`
}

// distanceMatrixOutput is the JSON shape we expect in the output line.
// For successful cases:
//
//	{"dist":[[0,1],[1,0]]}
//
// For error cases:
//
//	{"wantErr":"negative value"}
type distanceMatrixOutput struct {
	Dist    [][]float64 `json:"dist"`
	WantErr string      `json:"wantErr,omitempty"`
}

func almostEqual(a, b, tol float64) bool {
	return math.Abs(a-b) <= tol
}

// floatAlmostEqual2 kept for compatibility with older test code that calls
// floatAlmostEqual2; it behaves identically to almostEqual.
func floatAlmostEqual2(a, b, tol float64) bool {
	return math.Abs(a-b) <= tol
}

func TestDistanceMatrix_FromFile(t *testing.T) {
	const path = "clustering_tests/DistanceMatrix_tests.txt"

	cases, err := loadIOCasesFromFile(path)
	if err != nil {
		t.Fatalf("failed to load test cases from %q: %v", path, err)
	}

	if len(cases) == 0 {
		t.Fatalf("no test cases found in %q", path)
	}

	for _, tc := range cases {
		tc := tc // capture range variable
		t.Run(tc.Name, func(t *testing.T) {
			// Parse input JSON
			var in distanceMatrixInput
			if err := json.Unmarshal([]byte(tc.InputLine), &in); err != nil {
				t.Fatalf("failed to unmarshal input JSON %q: %v", tc.InputLine, err)
			}

			if len(in.Data) == 0 || len(in.Data[0]) == 0 {
				t.Fatalf("input data matrix must be non-empty")
			}

			// Convert [][]float64 → *mat.Dense
			r := len(in.Data)
			c := len(in.Data[0])
			flat := make([]float64, 0, r*c)
			for i := 0; i < r; i++ {
				if len(in.Data[i]) != c {
					t.Fatalf("row %d has length %d, expected %d", i, len(in.Data[i]), c)
				}
				flat = append(flat, in.Data[i]...)
			}
			data := mat.NewDense(r, c, flat)

			// Parse expected output JSON
			var out distanceMatrixOutput
			if err := json.Unmarshal([]byte(tc.OutputLine), &out); err != nil {
				t.Fatalf("failed to unmarshal output JSON %q: %v", tc.OutputLine, err)
			}

			// Run DistanceMatrix
			got := DistanceMatrix(data)

			// Handle error-expected cases (DistanceMatrix no longer returns errors, so skip these)
			if out.WantErr != "" {
				t.Skipf("Skipping error test case - DistanceMatrix no longer returns errors")
				return
			}

			// Compare dimensions
			gr, gc := got.Dims()
			if gr != len(out.Dist) {
				t.Fatalf("got %d rows, want %d", gr, len(out.Dist))
			}
			if gr == 0 {
				return
			}
			if gc != len(out.Dist[0]) {
				t.Fatalf("got %d cols, want %d", gc, len(out.Dist[0]))
			}
			// Compare values with a small tolerance
			for i := 0; i < gr; i++ {
				if len(out.Dist[i]) != gc {
					t.Fatalf("expected row %d to have %d columns, got %d", i, gc, len(out.Dist[i]))
				}
				for j := 0; j < gc; j++ {
					gotVal := got.At(i, j)
					wantVal := out.Dist[i][j]
					if !almostEqual(gotVal, wantVal, 1e-6) {
						t.Fatalf("dist[%d][%d] = %v, want %v", i, j, gotVal, wantVal)
					}
				}
			}
		})
	}
}

// parser for Euclidean
type knnWeightsInput struct {
	K    int         `json:"k"`
	Dist [][]float64 `json:"dist"`
}

// knnWeightsOutput matches the *output* JSON:
//
//	{"weights":[[0,0.5,0],[0.5,0,0],[0.3333333,0,0]]}
type knnWeightsOutput struct {
	Weights [][]float64 `json:"weights"`
}

// KNNWeightsCase is the fully parsed test case you can use in tests.
type KNNWeightsCase struct {
	Name   string
	Input  knnWeightsInput
	Output knnWeightsOutput
}

// loadKNNWeightsCases loads and fully parses all cases for
// fillDirectedKNNWeights from the given path.
func loadKNNWeightsCases(path string) ([]KNNWeightsCase, error) {
	rawCases, err := loadIOCasesFromFile(path)
	if err != nil {
		return nil, err
	}

	var result []KNNWeightsCase
	for _, rc := range rawCases {
		var in knnWeightsInput
		if err := json.Unmarshal([]byte(rc.InputLine), &in); err != nil {
			return nil, err
		}

		var out knnWeightsOutput
		if err := json.Unmarshal([]byte(rc.OutputLine), &out); err != nil {
			return nil, err
		}

		result = append(result, KNNWeightsCase{
			Name:   rc.Name,
			Input:  in,
			Output: out,
		})
	}

	return result, nil
}

// ---------- specialized structs for Euclidean ----------

// euclideanInput matches the *input* JSON:
//
//	{"first":[...], "second":[...]}
type euclideanInput struct {
	First  []float64 `json:"first"`
	Second []float64 `json:"second"`
}

// euclideanOutput matches the *output* JSON:
//
//	{"dist":5}
type euclideanOutput struct {
	Dist float64 `json:"dist"`
}

// EuclideanCase is the fully parsed test case for Euclidean().
type EuclideanCase struct {
	Name   string
	Input  euclideanInput
	Output euclideanOutput
}

// loadEuclideanCases loads & parses all Euclidean() cases from a file.
func loadEuclideanCases(path string) ([]EuclideanCase, error) {
	rawCases, err := loadIOCasesFromFile(path)
	if err != nil {
		return nil, err
	}

	var result []EuclideanCase
	for _, rc := range rawCases {
		var in euclideanInput
		if err := json.Unmarshal([]byte(rc.InputLine), &in); err != nil {
			return nil, err
		}

		var out euclideanOutput
		if err := json.Unmarshal([]byte(rc.OutputLine), &out); err != nil {
			return nil, err
		}

		result = append(result, EuclideanCase{
			Name:   rc.Name,
			Input:  in,
			Output: out,
		})
	}

	return result, nil
}

// test for Euclidean
func TestEuclidean_FromFile(t *testing.T) {
	const testFile = "clustering_tests/Euclidean.txt"

	cases, err := loadEuclideanCases(testFile)
	if err != nil {
		t.Fatalf("failed to load Euclidean cases from %s: %v", testFile, err)
	}

	almostEqual := func(a, b, tol float64) bool {
		return math.Abs(a-b) <= tol
	}

	for idx, tc := range cases {
		tc := tc
		name := fmt.Sprintf("case_%02d_%s", idx, tc.Name)

		t.Run(name, func(t *testing.T) {
			got := Euclidean(tc.Input.First, tc.Input.Second)
			want := tc.Output.Dist

			if !almostEqual(got, want, 1e-6) {
				t.Fatalf("Euclidean(%v, %v) = %v, want %v",
					tc.Input.First, tc.Input.Second, got, want)
			}
		})
	}
}

// ---------- parser for buildKNNGraph ----------

// knnGraphInput matches the *input* JSON:
//
//	{"k":1,"dist":[[0,1,2],[1,0,3],[2,3,0]]}
type knnGraphInput struct {
	K    int         `json:"k"`
	Dist [][]float64 `json:"dist"`
}

// expectedEdge is how we encode one edge in JSON:
//
//	{"to":1,"weight":1.0}
type expectedEdge struct {
	To     int     `json:"to"`
	Weight float64 `json:"weight"`
}

// knnGraphOutput matches the *expected* Graph:
//
//	{"nodes":3,"edges":{"0":[{"to":1,...}]}, "totalWeight":2.1666667}
type knnGraphOutput struct {
	Nodes       int                    `json:"nodes"`
	Edges       map[int][]expectedEdge `json:"edges"`
	TotalWeight float64                `json:"totalWeight"`
}

// BuildKNNGraphCase is the fully parsed test case.
type BuildKNNGraphCase struct {
	Name   string
	Input  knnGraphInput
	Output knnGraphOutput
}

// loadBuildKNNGraphCases loads & parses all BuildKNNGraph() cases from a file.
func loadBuildKNNGraphCases(path string) ([]BuildKNNGraphCase, error) {
	rawCases, err := loadIOCasesFromFile(path)
	if err != nil {
		return nil, err
	}

	var result []BuildKNNGraphCase
	for _, rc := range rawCases {
		var in knnGraphInput
		if err := json.Unmarshal([]byte(rc.InputLine), &in); err != nil {
			return nil, err
		}

		var out knnGraphOutput
		if err := json.Unmarshal([]byte(rc.OutputLine), &out); err != nil {
			return nil, err
		}

		result = append(result, BuildKNNGraphCase{
			Name:   rc.Name,
			Input:  in,
			Output: out,
		})
	}

	return result, nil
}

func TestBuildKNNGraph_FromFile(t *testing.T) {
	const testFile = "clustering_tests/buildKNNGraph.txt"

	cases, err := loadBuildKNNGraphCases(testFile)
	if err != nil {
		t.Fatalf("failed to load BuildKNNGraph cases from %s: %v", testFile, err)
	}

	almostEqual := func(a, b, tol float64) bool {
		return math.Abs(a-b) <= tol
	}

	for idx, tc := range cases {
		tc := tc
		name := fmt.Sprintf("case_%02d_%s", idx, tc.Name)

		t.Run(name, func(t *testing.T) {
			// ---------- build distance matrix from input ----------
			distVals := tc.Input.Dist
			rows := len(distVals)
			cols := 0
			if rows > 0 {
				cols = len(distVals[0])
			}

			var distMtx *mat.Dense

			if rows == 0 {
				// Special-case: empty distance matrix.
				// Use zero-value mat.Dense so Dims() returns (0,0) without panic.
				var d mat.Dense
				distMtx = &d
			} else {
				distMtx = mat.NewDense(rows, cols, nil)
				for i := 0; i < rows; i++ {
					if len(distVals[i]) != cols {
						t.Fatalf("row %d has length %d, expected %d", i, len(distVals[i]), cols)
					}
					for j := 0; j < cols; j++ {
						distMtx.Set(i, j, distVals[i][j])
					}
				}
			}

			// ---------- call function under test ----------
			got := BuildKNNGraph(distMtx, tc.Input.K)

			// ---------- check Nodes ----------
			if got.Nodes != tc.Output.Nodes {
				t.Fatalf("Nodes = %d, want %d", got.Nodes, tc.Output.Nodes)
			}

			// ---------- check TotalWeight ----------
			if !almostEqual(got.TotalWeight, tc.Output.TotalWeight, 1e-6) {
				t.Fatalf("TotalWeight = %v, want %v", got.TotalWeight, tc.Output.TotalWeight)
			}

			// ---------- check Edges (order-independent) ----------

			// expected: map[from][to]weight
			expEdgeMap := make(map[int]map[int]float64)
			for from, es := range tc.Output.Edges {
				if expEdgeMap[from] == nil {
					expEdgeMap[from] = make(map[int]float64)
				}
				for _, e := range es {
					expEdgeMap[from][e.To] = e.Weight
				}
			}

			// for each node in got, build got map[from][to]weight and compare
			for from := 0; from < got.Nodes; from++ {
				gotAdj := got.Edges[from]
				gotMap := make(map[int]float64, len(gotAdj))
				for _, e := range gotAdj {
					gotMap[e.To] = e.Weight
				}

				expMap, hasExp := expEdgeMap[from]
				if !hasExp {
					// if node wasn't in expected edges, it should have no outgoing edges
					if len(gotMap) != 0 {
						t.Fatalf("node %d: expected no outgoing edges, got %v", from, gotMap)
					}
					continue
				}

				if len(gotMap) != len(expMap) {
					t.Fatalf("node %d: got %d neighbors, want %d; got=%v, want=%v",
						from, len(gotMap), len(expMap), gotMap, expMap)
				}

				for to, wWant := range expMap {
					wGot, ok := gotMap[to]
					if !ok {
						t.Fatalf("node %d: missing edge to %d; got=%v, want edge with weight %v",
							from, to, gotMap, wWant)
					}
					if !almostEqual(wGot, wWant, 1e-6) {
						t.Fatalf("edge %d->%d: weight=%v, want %v", from, to, wGot, wWant)
					}
				}
			}
		})
	}
}

// ---------- parser for fillDirectedKNNWeight ----------

// ---- shared line-based parser for "# / JSON / @ / JSON" blocks ----

type fillDirRawCase struct {
	Name       string
	InputLine  string
	OutputLine string
}

func parseFillDirRawCases(path string) ([]fillDirRawCase, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var (
		cases            []fillDirRawCase
		currentName      string
		pendingInput     string
		waitingForInput  bool
		waitingForOutput bool
	)

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		switch {
		case strings.HasPrefix(line, "#"):
			// Start a new case
			currentName = strings.TrimSpace(strings.TrimPrefix(line, "#"))
			pendingInput = ""
			waitingForInput = true
			waitingForOutput = false

		case strings.HasPrefix(line, "@"):
			// Comment between input and output
			waitingForOutput = true

		default:
			if waitingForInput {
				// Input JSON line
				pendingInput = line
				waitingForInput = false
			} else if waitingForOutput {
				// Output JSON line → finalize case
				cases = append(cases, fillDirRawCase{
					Name:       currentName,
					InputLine:  pendingInput,
					OutputLine: line,
				})
				waitingForOutput = false
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return cases, nil
}

// ---- typed cases for fillDirectedKNNWeights ----

type fillDirInput struct {
	K    int         `json:"k"`
	Dist [][]float64 `json:"dist"`
}

type fillDirOutput struct {
	Weights [][]float64 `json:"weights"`
}

type FillDirectedKNNWeightsCase struct {
	Name   string
	Input  fillDirInput
	Output fillDirOutput
}

func loadFillDirectedKNNWeightsCases(path string) ([]FillDirectedKNNWeightsCase, error) {
	rawCases, err := parseFillDirRawCases(path)
	if err != nil {
		return nil, err
	}

	result := make([]FillDirectedKNNWeightsCase, 0, len(rawCases))
	for _, rc := range rawCases {
		var in fillDirInput
		if err := json.Unmarshal([]byte(rc.InputLine), &in); err != nil {
			return nil, err
		}

		var out fillDirOutput
		if err := json.Unmarshal([]byte(rc.OutputLine), &out); err != nil {
			return nil, err
		}

		result = append(result, FillDirectedKNNWeightsCase{
			Name:   rc.Name,
			Input:  in,
			Output: out,
		})
	}

	return result, nil
}

// ---------- parser for getNeighborsRow ----------

// One fully-parsed test case for getNeighborsRow.
type GetNeighborsRowCase struct {
	Name string
	Row  []float64
	R    int
	Idx  []int
	Dist []float64
}

// Internal raw representation of each block in the txt file.
type getNeighborsRowRawCase struct {
	Name       string
	InputLine  string
	OutputLine string
}

// loadGetNeighborsRowCases reads a file with blocks in this pattern:
//
//	# Case name
//	{"row":[...],"r":1}
//	@ some comment
//	{"idx":[...],"dist":[...]}
//
// and returns fully parsed cases.
func loadGetNeighborsRowCases(path string) ([]GetNeighborsRowCase, error) {
	rawCases, err := parseGetNeighborsRowRawCases(path)
	if err != nil {
		return nil, err
	}

	type input struct {
		Row []float64 `json:"row"`
		R   int       `json:"r"`
	}
	type output struct {
		Idx  []int     `json:"idx"`
		Dist []float64 `json:"dist"`
	}

	out := make([]GetNeighborsRowCase, 0, len(rawCases))
	for _, rc := range rawCases {
		var in input
		if err := json.Unmarshal([]byte(rc.InputLine), &in); err != nil {
			return nil, err
		}

		var exp output
		if err := json.Unmarshal([]byte(rc.OutputLine), &exp); err != nil {
			return nil, err
		}

		out = append(out, GetNeighborsRowCase{
			Name: rc.Name,
			Row:  in.Row,
			R:    in.R,
			Idx:  exp.Idx,
			Dist: exp.Dist,
		})
	}

	return out, nil
}

// parseGetNeighborsRowRawCases does the line-based parsing of
// "# / JSON / @ / JSON" into raw cases.
func parseGetNeighborsRowRawCases(path string) ([]getNeighborsRowRawCase, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var (
		cases            []getNeighborsRowRawCase
		currentName      string
		pendingInput     string
		waitingForInput  bool
		waitingForOutput bool
	)

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		switch {
		case strings.HasPrefix(line, "#"):
			// Start a new case
			currentName = strings.TrimSpace(strings.TrimPrefix(line, "#"))
			pendingInput = ""
			waitingForInput = true
			waitingForOutput = false

		case strings.HasPrefix(line, "@"):
			// Comment between input and output
			waitingForOutput = true

		default:
			if waitingForInput {
				// Input JSON
				pendingInput = line
				waitingForInput = false
			} else if waitingForOutput {
				// Output JSON → finalize case
				cases = append(cases, getNeighborsRowRawCase{
					Name:       currentName,
					InputLine:  pendingInput,
					OutputLine: line,
				})
				waitingForOutput = false
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return cases, nil
}

// test function for fillDirectedKNNWeights
func TestGetNeighborsRow_FromFile(t *testing.T) {
	const testFile = "clustering_tests/fillDirectedKNNWeights.txt"

	cases, err := loadGetNeighborsRowCases(testFile)
	if err != nil {
		t.Fatalf("failed to load getNeighborsRow cases from %s: %v", testFile, err)
	}

	almostEqual := func(a, b, tol float64) bool {
		return math.Abs(a-b) <= tol
	}

	for idx, tc := range cases {
		tc := tc
		name := fmt.Sprintf("getNeighbors_case_%02d_%s", idx, tc.Name)

		t.Run(name, func(t *testing.T) {
			neighbors := getNeighborsRow(tc.Row, tc.R)

			if len(neighbors) != len(tc.Idx) {
				t.Fatalf("len(neighbors) = %d, want %d", len(neighbors), len(tc.Idx))
			}

			for i, n := range neighbors {
				if n.Index != tc.Idx[i] {
					t.Fatalf("neighbor[%d].Index = %d, want %d", i, n.Index, tc.Idx[i])
				}
				if !almostEqual(n.Distance, tc.Dist[i], 1e-6) {
					t.Fatalf("neighbor[%d].Distance = %v, want %v", i, n.Distance, tc.Dist[i])
				}
			}
		})
	}
}

// ---------- parser for topKNeighbors ----------

// TopKNeighborsCase is one fully parsed test case for topKNeighbors.
type TopKNeighborsCase struct {
	Name       string
	K          int
	InputIdx   []int
	InputDist  []float64
	OutputIdx  []int
	OutputDist []float64
}

// loadTopKNeighborsCases reads a file with blocks like:
//
//	# Case name
//	{"k":2,"idx":[0,1,2],"dist":[5,1,3]}
//	@ some comment
//	{"idx":[1,2],"dist":[1,3]}
//
// and returns all parsed cases.
func loadTopKNeighborsCases(path string) ([]TopKNeighborsCase, error) {
	rawCases, err := parseTopKNeighborsRawCases(path)
	if err != nil {
		return nil, err
	}

	// JSON shapes for input/output
	type input struct {
		K    int       `json:"k"`
		Idx  []int     `json:"idx"`
		Dist []float64 `json:"dist"`
	}
	type output struct {
		Idx  []int     `json:"idx"`
		Dist []float64 `json:"dist"`
	}

	cases := make([]TopKNeighborsCase, 0, len(rawCases))
	for _, rc := range rawCases {
		var in input
		if err := json.Unmarshal([]byte(rc.InputLine), &in); err != nil {
			return nil, err
		}

		var out output
		if err := json.Unmarshal([]byte(rc.OutputLine), &out); err != nil {
			return nil, err
		}

		cases = append(cases, TopKNeighborsCase{
			Name:       rc.Name,
			K:          in.K,
			InputIdx:   in.Idx,
			InputDist:  in.Dist,
			OutputIdx:  out.Idx,
			OutputDist: out.Dist,
		})
	}

	return cases, nil
}

// ----- internal raw block parser: "# / JSON / @ / JSON" -----

type topKRawCase struct {
	Name       string
	InputLine  string
	OutputLine string
}

func parseTopKNeighborsRawCases(path string) ([]topKRawCase, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var (
		cases            []topKRawCase
		currentName      string
		pendingInput     string
		waitingForInput  bool
		waitingForOutput bool
	)

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		switch {
		case strings.HasPrefix(line, "#"):
			// start a new case
			currentName = strings.TrimSpace(strings.TrimPrefix(line, "#"))
			pendingInput = ""
			waitingForInput = true
			waitingForOutput = false

		case strings.HasPrefix(line, "@"):
			// comment between input and output
			waitingForOutput = true

		default:
			if waitingForInput {
				// this is the input JSON line
				pendingInput = line
				waitingForInput = false
			} else if waitingForOutput {
				// this is the output JSON line → finalize case
				cases = append(cases, topKRawCase{
					Name:       currentName,
					InputLine:  pendingInput,
					OutputLine: line,
				})
				waitingForOutput = false
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return cases, nil
}
func TestTopKNeighbors_FromFile(t *testing.T) {
	const testFile = "clustering_tests/topKNeighbors.txt"

	cases, err := loadTopKNeighborsCases(testFile)
	if err != nil {
		t.Fatalf("failed to load topKNeighbors cases from %s: %v", testFile, err)
	}

	almostEqual := func(a, b, tol float64) bool {
		return math.Abs(a-b) <= tol
	}

	for idx, tc := range cases {
		tc := tc
		name := fmt.Sprintf("topKNeighbors_case_%02d_%s", idx, tc.Name)

		t.Run(name, func(t *testing.T) {
			// sanity check: input idx/dist lengths must match
			if len(tc.InputIdx) != len(tc.InputDist) {
				t.Fatalf("input length mismatch: len(InputIdx) = %d, len(InputDist) = %d",
					len(tc.InputIdx), len(tc.InputDist))
			}

			// build []Neighbor from InputIdx/InputDist
			neighbors := make([]Neighbor, len(tc.InputIdx))
			for i := range tc.InputIdx {
				neighbors[i] = Neighbor{
					Index:    tc.InputIdx[i],
					Distance: tc.InputDist[i],
				}
			}

			// call function under test
			got := topKNeighbors(neighbors, tc.K)

			// expected lengths
			if len(got) != len(tc.OutputIdx) || len(got) != len(tc.OutputDist) {
				t.Fatalf("len(got) = %d, want %d (OutputIdx) and %d (OutputDist)",
					len(got), len(tc.OutputIdx), len(tc.OutputDist))
			}

			// element-wise check
			for i, n := range got {
				if n.Index != tc.OutputIdx[i] {
					t.Fatalf("got[%d].Index = %d, want %d", i, n.Index, tc.OutputIdx[i])
				}
				if !almostEqual(n.Distance, tc.OutputDist[i], 1e-6) {
					t.Fatalf("got[%d].Distance = %v, want %v", i, n.Distance, tc.OutputDist[i])
				}
			}
		})
	}
}

// ---------- parser for distanceToWeights ----------

// DistanceToWeightCase is one fully-parsed test case.
type DistanceToWeightCase struct {
	Name     string
	Distance float64
	Weight   float64
}

// loadDistanceToWeightCases reads a file with blocks like:
//
//	# Case name
//	{"distance": ...}
//	@ some comment
//	{"weight": ...}
//
// and returns all parsed cases.
func loadDistanceToWeightCases(path string) ([]DistanceToWeightCase, error) {
	rawCases, err := parseDistanceToWeightRawCases(path)
	if err != nil {
		return nil, err
	}

	type input struct {
		Distance float64 `json:"distance"`
	}
	type output struct {
		Weight float64 `json:"weight"`
	}

	cases := make([]DistanceToWeightCase, 0, len(rawCases))
	for _, rc := range rawCases {
		var in input
		if err := json.Unmarshal([]byte(rc.InputLine), &in); err != nil {
			return nil, err
		}

		var out output
		if err := json.Unmarshal([]byte(rc.OutputLine), &out); err != nil {
			return nil, err
		}

		cases = append(cases, DistanceToWeightCase{
			Name:     rc.Name,
			Distance: in.Distance,
			Weight:   out.Weight,
		})
	}

	return cases, nil
}

// ----- internal raw parser for "# / JSON / @ / JSON" -----

type dtwRawCase struct {
	Name       string
	InputLine  string
	OutputLine string
}

func parseDistanceToWeightRawCases(path string) ([]dtwRawCase, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var (
		cases            []dtwRawCase
		currentName      string
		pendingInput     string
		waitingForInput  bool
		waitingForOutput bool
	)

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		switch {
		case strings.HasPrefix(line, "#"):
			// start a new case
			currentName = strings.TrimSpace(strings.TrimPrefix(line, "#"))
			pendingInput = ""
			waitingForInput = true
			waitingForOutput = false

		case strings.HasPrefix(line, "@"):
			// comment line between input and output
			waitingForOutput = true

		default:
			if waitingForInput {
				// this is the input JSON line
				pendingInput = line
				waitingForInput = false
			} else if waitingForOutput {
				// this is the output JSON line → finalize case
				cases = append(cases, dtwRawCase{
					Name:       currentName,
					InputLine:  pendingInput,
					OutputLine: line,
				})
				waitingForOutput = false
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return cases, nil
}

func TestDistanceToWeight_FromFile(t *testing.T) {
	const testFile = "clustering_tests/distanceToWeights.txt"

	cases, err := loadDistanceToWeightCases(testFile)
	if err != nil {
		t.Fatalf("failed to load distanceToWeight cases from %s: %v", testFile, err)
	}

	almostEqual := func(a, b, tol float64) bool {
		return math.Abs(a-b) <= tol
	}

	for idx, tc := range cases {
		tc := tc
		name := fmt.Sprintf("distanceToWeight_case_%02d_%s", idx, tc.Name)

		t.Run(name, func(t *testing.T) {
			got := distanceToWeight(tc.Distance)
			want := tc.Weight

			// looser tolerance to account for rounding in the JSON (e.g. 0.2857143)
			if !almostEqual(got, want, 1e-6) {
				t.Fatalf("distanceToWeight(%v) = %v, want %v", tc.Distance, got, want)
			}
		})
	}
}

// ---------- parser for symmetricWeightUsing ----------

// ExpectedEdge is a lightweight version of your Edge for test expectations.
type ExpectedEdge struct {
	To     int     `json:"to"`
	Weight float64 `json:"weight"`
}

// SymmetrizeWeightsCase is one fully-parsed test case for symmetrizeWeightsUsing.
type SymmetrizeWeightsCase struct {
	Name        string
	Directed    [][]float64
	Edges       map[int][]ExpectedEdge
	TotalWeight float64
}

// loadSymmetrizeWeightsCases reads a file with blocks like:
//
//	# Case name
//	{"directed":[[...], ...]}
//	@ some comment
//	{"edges":{...},"totalWeight":...}
//
// and returns all parsed cases.
func loadSymmetrizeWeightsCases(path string) ([]SymmetrizeWeightsCase, error) {
	rawCases, err := parseSymmetrizeWeightsRawCases(path)
	if err != nil {
		return nil, err
	}

	// JSON shapes for input/output
	type input struct {
		Directed [][]float64 `json:"directed"`
	}
	type output struct {
		Edges       map[int][]ExpectedEdge `json:"edges"`
		TotalWeight float64                `json:"totalWeight"`
	}

	cases := make([]SymmetrizeWeightsCase, 0, len(rawCases))
	for _, rc := range rawCases {
		var in input
		if err := json.Unmarshal([]byte(rc.InputLine), &in); err != nil {
			return nil, err
		}

		var out output
		if err := json.Unmarshal([]byte(rc.OutputLine), &out); err != nil {
			return nil, err
		}

		cases = append(cases, SymmetrizeWeightsCase{
			Name:        rc.Name,
			Directed:    in.Directed,
			Edges:       out.Edges,
			TotalWeight: out.TotalWeight,
		})
	}

	return cases, nil
}

// ----- internal raw parser for "# / JSON / @ / JSON" -----

type symmRawCase struct {
	Name       string
	InputLine  string
	OutputLine string
}

func parseSymmetrizeWeightsRawCases(path string) ([]symmRawCase, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var (
		cases            []symmRawCase
		currentName      string
		pendingInput     string
		waitingForInput  bool
		waitingForOutput bool
	)

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		switch {
		case strings.HasPrefix(line, "#"):
			// start a new case
			currentName = strings.TrimSpace(strings.TrimPrefix(line, "#"))
			pendingInput = ""
			waitingForInput = true
			waitingForOutput = false

		case strings.HasPrefix(line, "@"):
			// comment between input and output
			waitingForOutput = true

		default:
			if waitingForInput {
				// this is the input JSON line
				pendingInput = line
				waitingForInput = false
			} else if waitingForOutput {
				// this is the output JSON line → finalize case
				cases = append(cases, symmRawCase{
					Name:       currentName,
					InputLine:  pendingInput,
					OutputLine: line,
				})
				waitingForOutput = false
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return cases, nil
}

func TestSymmetrizeWeightsUsing_FromFile(t *testing.T) {
	const testFile = "clustering_tests/symmetrizeWeightsUsing.txt"

	cases, err := loadSymmetrizeWeightsCases(testFile)
	if err != nil {
		t.Fatalf("failed to load symmetrizeWeightsUsing cases from %s: %v", testFile, err)
	}

	almostEqual := func(a, b, tol float64) bool {
		return math.Abs(a-b) <= tol
	}

	for idx, tc := range cases {
		tc := tc
		name := fmt.Sprintf("symmetrize_case_%02d_%s", idx, tc.Name)

		t.Run(name, func(t *testing.T) {
			// ---------- build directed *mat.Dense from tc.Directed ----------
			dvals := tc.Directed
			rows := len(dvals)
			cols := 0
			if rows > 0 {
				cols = len(dvals[0])
			}

			directed := mat.NewDense(rows, cols, nil)
			for i := 0; i < rows; i++ {
				if len(dvals[i]) != cols {
					t.Fatalf("row %d has length %d, expected %d", i, len(dvals[i]), cols)
				}
				for j := 0; j < cols; j++ {
					directed.Set(i, j, dvals[i][j])
				}
			}

			// ---------- call function under test ----------
			gotEdges, gotTotal := symmetrizeWeightsUsing(directed)

			// ---------- check TotalWeight ----------
			if !almostEqual(gotTotal, tc.TotalWeight, 1e-6) {
				t.Fatalf("TotalWeight = %v, want %v", gotTotal, tc.TotalWeight)
			}

			// ---------- check edges (order-independent) ----------

			// build expected map[from][to]weight
			expEdgeMap := make(map[int]map[int]float64)
			for from, es := range tc.Edges {
				if expEdgeMap[from] == nil {
					expEdgeMap[from] = make(map[int]float64)
				}
				for _, e := range es {
					expEdgeMap[from][e.To] = e.Weight
				}
			}

			// we know the number of nodes = rows of the directed matrix
			for from := 0; from < rows; from++ {
				gotAdj := gotEdges[from]
				gotMap := make(map[int]float64, len(gotAdj))
				for _, e := range gotAdj {
					gotMap[e.To] = e.Weight
				}

				expMap, hasExp := expEdgeMap[from]
				if !hasExp {
					// if not present in expected, we expect no outgoing edges
					if len(gotMap) != 0 {
						t.Fatalf("node %d: expected no edges, got %v", from, gotMap)
					}
					continue
				}

				if len(gotMap) != len(expMap) {
					t.Fatalf("node %d: got %d neighbors, want %d; got=%v, want=%v",
						from, len(gotMap), len(expMap), gotMap, expMap)
				}

				for to, wWant := range expMap {
					wGot, ok := gotMap[to]
					if !ok {
						t.Fatalf("node %d: missing edge to %d; got=%v, want edge weight %v",
							from, to, gotMap, wWant)
					}
					if !almostEqual(wGot, wWant, 1e-6) {
						t.Fatalf("edge %d->%d: weight=%v, want %v",
							from, to, wGot, wWant)
					}
				}
			}
		})
	}
}

// ---------- parser for Leiden----------

// EdgeJSON is a lightweight version of Edge for decoding from JSON.
type EdgeJSON struct {
	To     int     `json:"to"`
	Weight float64 `json:"weight"`
}

// LeidenCase is one fully parsed test case for Graph.Leiden.
type LeidenCase struct {
	Name       string
	Graph      *Graph
	Resolution float64
	Gamma      float64
	Theta      float64
	MaxIter    int
	Clusters   []int
}

// loadLeidenCases reads a file with blocks like:
//
//	# Case name
//	{"nodes":...,"edges":{...},"totalWeight":...,"resolution":...,"gamma":...,"theta":...,"maxIter":...}
//	@ some comment
//	{"clusters":[...]}
//
// and returns all parsed cases.
func loadLeidenCases(path string) ([]LeidenCase, error) {
	rawCases, err := parseLeidenRawCases(path)
	if err != nil {
		return nil, err
	}

	// JSON shapes for input / output
	type input struct {
		Nodes       int                `json:"nodes"`
		Edges       map[int][]EdgeJSON `json:"edges"`
		TotalWeight float64            `json:"totalWeight"`
		Resolution  float64            `json:"resolution"`
		Gamma       float64            `json:"gamma"`
		Theta       float64            `json:"theta"`
		MaxIter     int                `json:"maxIter"`
	}
	type output struct {
		Clusters []int `json:"clusters"`
	}

	cases := make([]LeidenCase, 0, len(rawCases))
	for _, rc := range rawCases {
		var in input
		if err := json.Unmarshal([]byte(rc.InputLine), &in); err != nil {
			return nil, err
		}

		var out output
		if err := json.Unmarshal([]byte(rc.OutputLine), &out); err != nil {
			return nil, err
		}

		// convert EdgeJSON → Edge
		edges := make(map[int][]Edge, len(in.Edges))
		for from, ejs := range in.Edges {
			es := make([]Edge, len(ejs))
			for i, ej := range ejs {
				es[i] = Edge{
					To:     ej.To,
					Weight: ej.Weight,
				}
			}
			edges[from] = es
		}

		g := &Graph{
			Nodes:       in.Nodes,
			Edges:       edges,
			TotalWeight: in.TotalWeight,
		}

		cases = append(cases, LeidenCase{
			Name:       rc.Name,
			Graph:      g,
			Resolution: in.Resolution,
			Gamma:      in.Gamma,
			Theta:      in.Theta,
			MaxIter:    in.MaxIter,
			Clusters:   out.Clusters,
		})
	}

	return cases, nil
}

// ---------- internal raw parser for "# / JSON / @ / JSON" ----------

type leidenRawCase struct {
	Name       string
	InputLine  string
	OutputLine string
}

func parseLeidenRawCases(path string) ([]leidenRawCase, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var (
		cases            []leidenRawCase
		currentName      string
		pendingInput     string
		waitingForInput  bool
		waitingForOutput bool
	)

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		switch {
		case strings.HasPrefix(line, "#"):
			// start a new case
			currentName = strings.TrimSpace(strings.TrimPrefix(line, "#"))
			pendingInput = ""
			waitingForInput = true
			waitingForOutput = false

		case strings.HasPrefix(line, "@"):
			// comment between input and output
			waitingForOutput = true

		default:
			if waitingForInput {
				// this is the input JSON line
				pendingInput = line
				waitingForInput = false
			} else if waitingForOutput {
				// this is the output JSON line → finalize case
				cases = append(cases, leidenRawCase{
					Name:       currentName,
					InputLine:  pendingInput,
					OutputLine: line,
				})
				waitingForOutput = false
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return cases, nil
}

func TestLeiden_FromFile(t *testing.T) {
	const testFile = "clustering_tests/Leiden.txt"

	cases, err := loadLeidenCases(testFile)
	if err != nil {
		t.Fatalf("loadLeidenCases(%q) error: %v", testFile, err)
	}
	if len(cases) == 0 {
		t.Fatalf("no Leiden test cases loaded from %q", testFile)
	}

	for idx, tc := range cases {
		name := fmt.Sprintf("case_%02d_%s", idx, strings.TrimSpace(tc.Name))

		t.Run(name, func(t *testing.T) {
			got := tc.Graph.Leiden(tc.Resolution, tc.Gamma, tc.Theta, tc.MaxIter)

			if len(got) != len(tc.Clusters) {
				t.Fatalf("clusters length mismatch: got %d, want %d; got=%v, want=%v",
					len(got), len(tc.Clusters), got, tc.Clusters)
			}

			for i := range got {
				if got[i] != tc.Clusters[i] {
					t.Fatalf("clusters differ at index %d: got %v, want %v",
						i, got, tc.Clusters)
				}
			}
		})
	}
}

// ---------- parser for Refine ----------

// RefineCase is one fully parsed test case for Graph.Refine.
type RefineCase struct {
	Name      string
	Partition []int
	Clusters  []int
}

// loadRefineCases reads a file with blocks:
//
//	# Case name
//	{"partition":[...]}
//	@ some comment
//	{"clusters":[...]}
//
// and returns all parsed cases.
func loadRefineCases(path string) ([]RefineCase, error) {
	rawCases, err := parseRefineRawCases(path)
	if err != nil {
		return nil, err
	}

	// JSON shapes for input/output
	type input struct {
		Partition []int `json:"partition"`
	}
	type output struct {
		Clusters []int `json:"clusters"`
	}

	cases := make([]RefineCase, 0, len(rawCases))
	for _, rc := range rawCases {
		var in input
		if err := json.Unmarshal([]byte(rc.InputLine), &in); err != nil {
			return nil, err
		}

		var out output
		if err := json.Unmarshal([]byte(rc.OutputLine), &out); err != nil {
			return nil, err
		}

		cases = append(cases, RefineCase{
			Name:      rc.Name,
			Partition: in.Partition,
			Clusters:  out.Clusters,
		})
	}

	return cases, nil
}

// -------- internal raw parser for "# / JSON / @ / JSON" --------

type refineRawCase struct {
	Name       string
	InputLine  string
	OutputLine string
}

func parseRefineRawCases(path string) ([]refineRawCase, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var (
		cases            []refineRawCase
		currentName      string
		pendingInput     string
		waitingForInput  bool
		waitingForOutput bool
	)

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		switch {
		case strings.HasPrefix(line, "#"):
			// start a new case
			currentName = strings.TrimSpace(strings.TrimPrefix(line, "#"))
			pendingInput = ""
			waitingForInput = true
			waitingForOutput = false

		case strings.HasPrefix(line, "@"):
			// comment between input and output
			waitingForOutput = true

		default:
			if waitingForInput {
				// input JSON line
				pendingInput = line
				waitingForInput = false
			} else if waitingForOutput {
				// output JSON line → finalize case
				cases = append(cases, refineRawCase{
					Name:       currentName,
					InputLine:  pendingInput,
					OutputLine: line,
				})
				waitingForOutput = false
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return cases, nil
}

func TestRefine_FromFile(t *testing.T) {
	const testFile = "clustering_tests/Refine.txt" // adjust if your path is different

	cases, err := loadRefineCases(testFile)
	if err != nil {
		t.Fatalf("loadRefineCases(%q) error: %v", testFile, err)
	}
	if len(cases) == 0 {
		t.Fatalf("no Refine test cases loaded from %q", testFile)
	}

	for idx, tc := range cases {
		name := fmt.Sprintf("case_%02d_%s", idx, strings.TrimSpace(tc.Name))

		t.Run(name, func(t *testing.T) {
			// If Refine is a method on Graph, like:
			//     func (g *Graph) Refine(partition []int) []int
			// we can just use an empty graph here, since your test
			// cases only depend on the partition, not on g.
			var g Graph
			got := g.Refine(tc.Partition)

			if len(got) != len(tc.Clusters) {
				t.Fatalf("length mismatch: got %d, want %d; got=%v, want=%v",
					len(got), len(tc.Clusters), got, tc.Clusters)
			}

			for i := range got {
				if got[i] != tc.Clusters[i] {
					t.Fatalf("clusters differ at index %d: got %v, want %v",
						i, got, tc.Clusters)
				}
			}
		})
	}
}

// parser for RefinePartition

// RefinePartitionCase is one fully parsed test case for Graph.RefinePartition.
type RefinePartitionCase struct {
	Name       string
	Graph      *Graph
	Partition  []int
	Resolution float64
	Gamma      float64
	Theta      float64
	Clusters   []int
}

// loadRefinePartitionCases reads a file with blocks:
//
//	# Case name
//	{"nodes":...,"edges":{...},"totalWeight":...,"partition":[...],
//	 "resolution":...,"gamma":...,"theta":...}
//	@ some comment
//	{"clusters":[...]}
//
// and returns all parsed cases.
func loadRefinePartitionCases(path string) ([]RefinePartitionCase, error) {
	rawCases, err := parseRefinePartitionRawCases(path)
	if err != nil {
		return nil, err
	}

	// JSON shapes for input/output
	type input struct {
		Nodes       int                `json:"nodes"`
		Edges       map[int][]EdgeJSON `json:"edges"`
		TotalWeight float64            `json:"totalWeight"`
		Partition   []int              `json:"partition"`
		Resolution  float64            `json:"resolution"`
		Gamma       float64            `json:"gamma"`
		Theta       float64            `json:"theta"`
	}
	type output struct {
		Clusters []int `json:"clusters"`
	}

	cases := make([]RefinePartitionCase, 0, len(rawCases))
	for _, rc := range rawCases {
		var in input
		if err := json.Unmarshal([]byte(rc.InputLine), &in); err != nil {
			return nil, err
		}

		var out output
		if err := json.Unmarshal([]byte(rc.OutputLine), &out); err != nil {
			return nil, err
		}

		// convert EdgeJSON → Edge
		edges := make(map[int][]Edge, len(in.Edges))
		for from, ejs := range in.Edges {
			es := make([]Edge, len(ejs))
			for i, ej := range ejs {
				es[i] = Edge{
					To:     ej.To,
					Weight: ej.Weight,
				}
			}
			edges[from] = es
		}

		g := &Graph{
			Nodes:       in.Nodes,
			Edges:       edges,
			TotalWeight: in.TotalWeight,
		}

		cases = append(cases, RefinePartitionCase{
			Name:       rc.Name,
			Graph:      g,
			Partition:  in.Partition,
			Resolution: in.Resolution,
			Gamma:      in.Gamma,
			Theta:      in.Theta,
			Clusters:   out.Clusters,
		})
	}

	return cases, nil
}

// ---------- internal raw parser for "# / JSON / @ / JSON" ----------

type refinePartitionRawCase struct {
	Name       string
	InputLine  string
	OutputLine string
}

func parseRefinePartitionRawCases(path string) ([]refinePartitionRawCase, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var (
		cases            []refinePartitionRawCase
		currentName      string
		pendingInput     string
		waitingForInput  bool
		waitingForOutput bool
	)

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		switch {
		case strings.HasPrefix(line, "#"):
			// start a new case
			currentName = strings.TrimSpace(strings.TrimPrefix(line, "#"))
			pendingInput = ""
			waitingForInput = true
			waitingForOutput = false

		case strings.HasPrefix(line, "@"):
			// comment between input and output
			waitingForOutput = true

		default:
			if waitingForInput {
				// this is the input JSON line
				pendingInput = line
				waitingForInput = false
			} else if waitingForOutput {
				// this is the output JSON line → finalize case
				cases = append(cases, refinePartitionRawCase{
					Name:       currentName,
					InputLine:  pendingInput,
					OutputLine: line,
				})
				waitingForOutput = false
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return cases, nil
}

// ---------- parser for RefinePartition ----------

// EdgeJSON is a lightweight version of Edge for decoding from JSON.

// loadRefinePartitionCases reads a file with blocks:
//
//   # Case name
//   {"nodes":...,"edges":{...},"totalWeight":...,"partition":[...],
//    "resolution":...,"gamma":...,"theta":...}
//   @ some comment
//   {"clusters":[...]}
//
// and returns all parsed cases.

// JSON shapes for input/output.
type input struct {
	Nodes       int                `json:"nodes"`
	Edges       map[int][]EdgeJSON `json:"edges"`
	TotalWeight float64            `json:"totalWeight"`
	Partition   []int              `json:"partition"`
	Resolution  float64            `json:"resolution"`
	Gamma       float64            `json:"gamma"`
	Theta       float64            `json:"theta"`
}
type output struct {
	Clusters []int `json:"clusters"`
}

func TestRefinePartition_FromFile(t *testing.T) {
	const testFile = "clustering_tests/RefinePartition.txt" // adjust if needed

	cases, err := loadRefinePartitionCases(testFile)
	if err != nil {
		t.Fatalf("loadRefinePartitionCases(%q) error: %v", testFile, err)
	}
	if len(cases) == 0 {
		t.Fatalf("no RefinePartition test cases loaded from %q", testFile)
	}

	for idx, tc := range cases {
		name := fmt.Sprintf("case_%02d_%s", idx, strings.TrimSpace(tc.Name))

		t.Run(name, func(t *testing.T) {
			// Assuming your method signature is:
			// func (g *Graph) RefinePartition(partition []int, resolution, gamma, theta float64) []int
			got := tc.Graph.RefinePartition(tc.Partition, tc.Resolution, tc.Gamma, tc.Theta, 10)

			if len(got) != len(tc.Clusters) {
				t.Fatalf("length mismatch: got %d, want %d; got=%v, want=%v",
					len(got), len(tc.Clusters), got, tc.Clusters)
			}

			for i := range got {
				if got[i] != tc.Clusters[i] {
					t.Fatalf("clusters differ at index %d: got %v, want %v",
						i, got, tc.Clusters)
				}
			}
		})
	}
}

// parser for InitSingletonPartition

// InitSingletonPartitionCase is one fully parsed test case.
type InitSingletonPartitionCase struct {
	Name      string
	N         int
	Partition []int
}

// loadInitSingletonPartitionCases reads a file with blocks:
//
//	# Case name
//	{"n":...}
//	@ some comment
//	{"partition":[...]}
//
// and returns all parsed cases.
func loadInitSingletonPartitionCases(path string) ([]InitSingletonPartitionCase, error) {
	rawCases, err := parseInitSingletonPartitionRawCases(path)
	if err != nil {
		return nil, err
	}

	// JSON shapes for input/output.
	type input struct {
		N int `json:"n"`
	}
	type output struct {
		Partition []int `json:"partition"`
	}

	cases := make([]InitSingletonPartitionCase, 0, len(rawCases))
	for _, rc := range rawCases {
		var in input
		if err := json.Unmarshal([]byte(rc.InputLine), &in); err != nil {
			return nil, err
		}

		var out output
		if err := json.Unmarshal([]byte(rc.OutputLine), &out); err != nil {
			return nil, err
		}

		cases = append(cases, InitSingletonPartitionCase{
			Name:      rc.Name,
			N:         in.N,
			Partition: out.Partition,
		})
	}

	return cases, nil
}

// ---------- internal raw parser for "# / JSON / @ / JSON" ----------

type initSingletonPartitionRawCase struct {
	Name       string
	InputLine  string
	OutputLine string
}

func parseInitSingletonPartitionRawCases(path string) ([]initSingletonPartitionRawCase, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var (
		cases            []initSingletonPartitionRawCase
		currentName      string
		pendingInput     string
		waitingForInput  bool
		waitingForOutput bool
	)

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		switch {
		case strings.HasPrefix(line, "#"):
			// start a new case
			currentName = strings.TrimSpace(strings.TrimPrefix(line, "#"))
			pendingInput = ""
			waitingForInput = true
			waitingForOutput = false

		case strings.HasPrefix(line, "@"):
			// comment between input and output
			waitingForOutput = true

		default:
			if waitingForInput {
				// this is the input JSON line
				pendingInput = line
				waitingForInput = false
			} else if waitingForOutput {
				// this is the output JSON line → finalize case
				cases = append(cases, initSingletonPartitionRawCase{
					Name:       currentName,
					InputLine:  pendingInput,
					OutputLine: line,
				})
				waitingForOutput = false
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return cases, nil
}

// parser for QL_InitSingletonPartition

// EdgeJSON is a lightweight version of Edge for JSON decoding.

// GraphInitSingletonPartitionCase is one fully parsed test case
// for (*Graph).InitSingletonPartition().
type GraphInitSingletonPartitionCase struct {
	Name      string
	Graph     *Graph
	Partition []int
}

// loadGraphInitSingletonPartitionCases reads a file with blocks:
//
//	# Case name
//	{"nodes":...,"edges":{...},"totalWeight":...}
//	@ some comment
//	{"partition":[...]}
//
// and returns all parsed cases.
func loadGraphInitSingletonPartitionCases(path string) ([]GraphInitSingletonPartitionCase, error) {
	rawCases, err := parseGraphInitSingletonPartitionRawCases(path)
	if err != nil {
		return nil, err
	}

	// JSON shapes for input/output.
	type input struct {
		Nodes       int                `json:"nodes"`
		Edges       map[int][]EdgeJSON `json:"edges"`
		TotalWeight float64            `json:"totalWeight"`
	}
	type output struct {
		Partition []int `json:"partition"`
	}

	cases := make([]GraphInitSingletonPartitionCase, 0, len(rawCases))
	for _, rc := range rawCases {
		var in input
		if err := json.Unmarshal([]byte(rc.InputLine), &in); err != nil {
			return nil, err
		}

		var out output
		if err := json.Unmarshal([]byte(rc.OutputLine), &out); err != nil {
			return nil, err
		}

		// convert EdgeJSON → Edge to build *Graph
		edges := make(map[int][]Edge, len(in.Edges))
		for from, ejs := range in.Edges {
			es := make([]Edge, len(ejs))
			for i, ej := range ejs {
				es[i] = Edge{
					To:     ej.To,
					Weight: ej.Weight,
				}
			}
			edges[from] = es
		}

		g := &Graph{
			Nodes:       in.Nodes,
			Edges:       edges,
			TotalWeight: in.TotalWeight,
		}

		cases = append(cases, GraphInitSingletonPartitionCase{
			Name:      rc.Name,
			Graph:     g,
			Partition: out.Partition,
		})
	}

	return cases, nil
}

// ---------- internal raw parser for "# / JSON / @ / JSON" ----------

type graphInitSingletonPartitionRawCase struct {
	Name       string
	InputLine  string
	OutputLine string
}

func parseGraphInitSingletonPartitionRawCases(path string) ([]graphInitSingletonPartitionRawCase, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var (
		cases            []graphInitSingletonPartitionRawCase
		currentName      string
		pendingInput     string
		waitingForInput  bool
		waitingForOutput bool
	)

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		switch {
		case strings.HasPrefix(line, "#"):
			// start a new case
			currentName = strings.TrimSpace(strings.TrimPrefix(line, "#"))
			pendingInput = ""
			waitingForInput = true
			waitingForOutput = false

		case strings.HasPrefix(line, "@"):
			// comment between input and output
			waitingForOutput = true

		default:
			if waitingForInput {
				// this is the input JSON line
				pendingInput = line
				waitingForInput = false
			} else if waitingForOutput {
				// this is the output JSON line → finalize case
				cases = append(cases, graphInitSingletonPartitionRawCase{
					Name:       currentName,
					InputLine:  pendingInput,
					OutputLine: line,
				})
				waitingForOutput = false
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return cases, nil
}

func TestInitSingletonPartition_FromFile(t *testing.T) {
	const testFile = "clustering_tests/InitSingletonPartition.txt" // adjust path if needed

	cases, err := loadInitSingletonPartitionCases(testFile)
	if err != nil {
		t.Fatalf("loadInitSingletonPartitionCases(%q) error: %v", testFile, err)
	}
	if len(cases) == 0 {
		t.Fatalf("no InitSingletonPartition test cases loaded from %q", testFile)
	}

	for idx, tc := range cases {
		name := fmt.Sprintf("case_%02d_%s", idx, strings.TrimSpace(tc.Name))

		t.Run(name, func(t *testing.T) {
			got := InitSingletonPartition(tc.N)

			if len(got) != len(tc.Partition) {
				t.Fatalf("length mismatch: got %d, want %d; got=%v, want=%v",
					len(got), len(tc.Partition), got, tc.Partition)
			}

			for i := range got {
				if got[i] != tc.Partition[i] {
					t.Fatalf("partition differs at index %d: got %v, want %v",
						i, got, tc.Partition)
				}
			}
		})
	}
}

// parser for NodesByCluster

// NodesByClusterCase is one fully parsed test case for NodesByCluster.
type NodesByClusterCase struct {
	Name      string
	Partition []int
	Groups    map[int][]int
}

// loadNodesByClusterCases reads a file with blocks:
//
//	# Case name
//	{"partition":[...]}
//	@ some comment
//	{"groups":{"0":[...], "1":[...]}}
//
// and returns all parsed cases.
func loadNodesByClusterCases(path string) ([]NodesByClusterCase, error) {
	rawCases, err := parseNodesByClusterRawCases(path)
	if err != nil {
		return nil, err
	}

	// JSON shapes for input/output.
	type input struct {
		Partition []int `json:"partition"`
	}
	type output struct {
		Groups map[int][]int `json:"groups"`
	}

	cases := make([]NodesByClusterCase, 0, len(rawCases))
	for _, rc := range rawCases {
		var in input
		if err := json.Unmarshal([]byte(rc.InputLine), &in); err != nil {
			return nil, err
		}

		var out output
		if err := json.Unmarshal([]byte(rc.OutputLine), &out); err != nil {
			return nil, err
		}

		cases = append(cases, NodesByClusterCase{
			Name:      rc.Name,
			Partition: in.Partition,
			Groups:    out.Groups,
		})
	}

	return cases, nil
}

// ---------- internal raw parser for "# / JSON / @ / JSON" ----------

type nodesByClusterRawCase struct {
	Name       string
	InputLine  string
	OutputLine string
}

func parseNodesByClusterRawCases(path string) ([]nodesByClusterRawCase, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var (
		cases            []nodesByClusterRawCase
		currentName      string
		pendingInput     string
		waitingForInput  bool
		waitingForOutput bool
	)

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		switch {
		case strings.HasPrefix(line, "#"):
			// start a new case
			currentName = strings.TrimSpace(strings.TrimPrefix(line, "#"))
			pendingInput = ""
			waitingForInput = true
			waitingForOutput = false

		case strings.HasPrefix(line, "@"):
			// comment between input and output
			waitingForOutput = true

		default:
			if waitingForInput {
				// this is the input JSON line
				pendingInput = line
				waitingForInput = false
			} else if waitingForOutput {
				// this is the output JSON line → finalize case
				cases = append(cases, nodesByClusterRawCase{
					Name:       currentName,
					InputLine:  pendingInput,
					OutputLine: line,
				})
				waitingForOutput = false
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return cases, nil
}

func TestNodesByCluster_FromFile(t *testing.T) {
	const testFile = "clustering_tests/NodesByCluster.txt" // adjust path if needed

	cases, err := loadNodesByClusterCases(testFile)
	if err != nil {
		t.Fatalf("loadNodesByClusterCases(%q) error: %v", testFile, err)
	}
	if len(cases) == 0 {
		t.Fatalf("no NodesByCluster test cases loaded from %q", testFile)
	}

	for idx, tc := range cases {
		name := fmt.Sprintf("case_%02d_%s", idx, strings.TrimSpace(tc.Name))

		t.Run(name, func(t *testing.T) {
			// Graph isn't actually needed by NodesByCluster, so an empty one is fine.
			var g Graph
			got := g.NodesByCluster(tc.Partition)

			// Compare map sizes
			if len(got) != len(tc.Groups) {
				t.Fatalf("group count mismatch: got %d, want %d; got=%v, want=%v",
					len(got), len(tc.Groups), got, tc.Groups)
			}

			// Compare each group
			for cid, wantNodes := range tc.Groups {
				gotNodes, ok := got[cid]
				if !ok {
					t.Fatalf("missing cluster %d in result; got=%v, want=%v", cid, got, tc.Groups)
				}
				if len(gotNodes) != len(wantNodes) {
					t.Fatalf("cluster %d size mismatch: got %d, want %d; got=%v, want=%v",
						cid, len(gotNodes), len(wantNodes), gotNodes, wantNodes)
				}
				for i := range gotNodes {
					if gotNodes[i] != wantNodes[i] {
						t.Fatalf("cluster %d nodes differ at index %d: got %v, want %v",
							cid, i, gotNodes, wantNodes)
					}
				}
			}
		})
	}
}

//parser for MergeNodesSubset

// MergeNodesSubsetCase is one fully parsed test case for Graph.MergeNodesSubset.
type MergeNodesSubsetCase struct {
	Name             string
	Nodes            []int
	Partition        []int
	RefinedPartition []int
	Resolution       float64
	Gamma            float64
	Theta            float64
	Expected         []int
}

// loadMergeNodesSubsetCases reads a file with blocks:
//
//	# Case name
//	{"nodes":[...],"partition":[...],"refinedPartition":[...],
//	 "resolution":...,"gamma":...,"theta":...}
//	@ some comment
//	{"partition":[...]}    // expected refined partition
//
// and returns all parsed cases.
func loadMergeNodesSubsetCases(path string) ([]MergeNodesSubsetCase, error) {
	rawCases, err := parseMergeNodesSubsetRawCases(path)
	if err != nil {
		return nil, err
	}

	// JSON shapes for input/output.
	type input struct {
		Nodes            []int   `json:"nodes"`
		Partition        []int   `json:"partition"`
		RefinedPartition []int   `json:"refinedPartition"`
		Resolution       float64 `json:"resolution"`
		Gamma            float64 `json:"gamma"`
		Theta            float64 `json:"theta"`
	}
	type output struct {
		Partition []int `json:"partition"`
	}

	cases := make([]MergeNodesSubsetCase, 0, len(rawCases))
	for _, rc := range rawCases {
		var in input
		if err := json.Unmarshal([]byte(rc.InputLine), &in); err != nil {
			return nil, err
		}

		var out output
		if err := json.Unmarshal([]byte(rc.OutputLine), &out); err != nil {
			return nil, err
		}

		cases = append(cases, MergeNodesSubsetCase{
			Name:             rc.Name,
			Nodes:            in.Nodes,
			Partition:        in.Partition,
			RefinedPartition: in.RefinedPartition,
			Resolution:       in.Resolution,
			Gamma:            in.Gamma,
			Theta:            in.Theta,
			Expected:         out.Partition,
		})
	}

	return cases, nil
}

// ---------- internal raw parser for "# / JSON / @ / JSON" ----------

type mergeNodesSubsetRawCase struct {
	Name       string
	InputLine  string
	OutputLine string
}

func parseMergeNodesSubsetRawCases(path string) ([]mergeNodesSubsetRawCase, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var (
		cases            []mergeNodesSubsetRawCase
		currentName      string
		pendingInput     string
		waitingForInput  bool
		waitingForOutput bool
	)

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		switch {
		case strings.HasPrefix(line, "#"):
			// start a new case
			currentName = strings.TrimSpace(strings.TrimPrefix(line, "#"))
			pendingInput = ""
			waitingForInput = true
			waitingForOutput = false

		case strings.HasPrefix(line, "@"):
			// comment between input and output
			waitingForOutput = true

		default:
			if waitingForInput {
				// this is the input JSON line
				pendingInput = line
				waitingForInput = false
			} else if waitingForOutput {
				// this is the output JSON line → finalize case
				cases = append(cases, mergeNodesSubsetRawCase{
					Name:       currentName,
					InputLine:  pendingInput,
					OutputLine: line,
				})
				waitingForOutput = false
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return cases, nil
}

func TestMergeNodesSubset_FromFile(t *testing.T) {
	const testFile = "clustering_tests/MergeNodesSubset.txt" // put those 4 cases here

	cases, err := loadMergeNodesSubsetCases(testFile)
	if err != nil {
		t.Fatalf("loadMergeNodesSubsetCases(%q) error: %v", testFile, err)
	}
	if len(cases) == 0 {
		t.Fatalf("no MergeNodesSubset test cases loaded from %q", testFile)
	}

	for idx, tc := range cases {
		name := fmt.Sprintf("case_%02d_%s", idx, strings.TrimSpace(tc.Name))

		t.Run(name, func(t *testing.T) {
			var g Graph

			// If your method is named MergeNodes instead of MergeNodesSubset,
			// just change this call accordingly.
			// Build nodesByCluster map from the partition
			nodesByCluster := g.NodesByCluster(tc.Partition)
			maxRounds := 10 // default max rounds for testing
			got := g.MergeNodesSubset(
				tc.Nodes,
				tc.Partition,
				tc.RefinedPartition,
				tc.Resolution,
				tc.Gamma,
				tc.Theta,
				maxRounds,
				nodesByCluster,
			)

			if len(got) != len(tc.Expected) {
				t.Fatalf("length mismatch: got %d, want %d; got=%v, want=%v",
					len(got), len(tc.Expected), got, tc.Expected)
			}

			for i := range got {
				if got[i] != tc.Expected[i] {
					t.Fatalf("partition differs at index %d: got %v, want %v",
						i, got, tc.Expected)
				}
			}
		})
	}
}

// parser for FindWellConnectedClusters

// -----------------------------------------------------------------------------
// Helpers (you can reuse existing ones if you already have them)
// -----------------------------------------------------------------------------

// tcName2 generates a nice subtest name like "case_01_name".
func tcName2(idx int, name string) string {
	base := fmt.Sprintf("case_%02d", idx+1)
	name = strings.TrimSpace(name)
	if name == "" {
		return base
	}
	name = strings.ReplaceAll(name, " ", "_")
	return base + "_" + name
}

// equalIntSlicesIgnoringOrder compares two int slices as sets (ignores order).
func equalIntSlicesIgnoringOrder(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	aa := append([]int(nil), a...)
	bb := append([]int(nil), b...)
	sort.Ints(aa)
	sort.Ints(bb)
	for i := range aa {
		if aa[i] != bb[i] {
			return false
		}
	}
	return true
}

// buildGraphFromDirected constructs a *Graph from a symmetric adjacency matrix
// where directed[i][j] is the weight from node i to node j.
//
// IMPORTANT: adapt this to your actual Graph / Edge types.
func buildGraphFromDirected(directed [][]float64) *Graph {
	g := &Graph{}

	// Example if you have:
	//   type Edge struct { To int; Weight float64 }
	//   type Graph struct { Edges map[int][]Edge }
	//
	// g.Edges = make(map[int][]Edge)
	// for i := range directed {
	//     for j, w := range directed[i] {
	//         if w == 0 {
	//             continue
	//         }
	//         g.Edges[i] = append(g.Edges[i], Edge{To: j, Weight: w})
	//     }
	// }
	//
	// return g

	return g
}

// -----------------------------------------------------------------------------
// Parser for FindWellConnectedClusters test cases
//

//
//   # Case name
//   {"directed":[[...]],"subset":[...],"refinedPartition":[...],"gamma":0.5}
//   @ some comment
//   {"candidates":[...]}
// -----------------------------------------------------------------------------

// FindWellConnectedClustersRawCase is one raw block from the text file.
type FindWellConnectedClustersRawCase struct {
	Name       string
	InputLine  string
	OutputLine string
}

// parseFindWellConnectedClustersRawCases reads the text file and returns all
// raw cases (name + input JSON line + output JSON line).
func parseFindWellConnectedClustersRawCases(path string) ([]FindWellConnectedClustersRawCase, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)

	var (
		cases                             []FindWellConnectedClustersRawCase
		currentName                       string
		inputLine, outputLine             string
		waitingForInput, waitingForOutput bool
		lineNum                           int
	)

	flushCase := func() error {
		if inputLine == "" && outputLine == "" {
			return nil
		}
		if inputLine == "" || outputLine == "" {
			return fmt.Errorf("incomplete case %q: missing input or output", currentName)
		}
		name := strings.TrimSpace(currentName)
		if name == "" {
			name = fmt.Sprintf("unnamed_%d", len(cases)+1)
		}
		cases = append(cases, FindWellConnectedClustersRawCase{
			Name:       name,
			InputLine:  inputLine,
			OutputLine: outputLine,
		})
		currentName = ""
		inputLine, outputLine = "", ""
		waitingForInput, waitingForOutput = false, false
		return nil
	}

	for scanner.Scan() {
		lineNum++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		switch {
		case strings.HasPrefix(line, "#"):
			// New case header – flush previous case (if any).
			if err := flushCase(); err != nil {
				return nil, fmt.Errorf("line %d: %w", lineNum, err)
			}
			currentName = strings.TrimSpace(strings.TrimPrefix(line, "#"))
			waitingForInput, waitingForOutput = true, false

		case strings.HasPrefix(line, "@"):
			// Comment line – next JSON line is the output.
			waitingForInput, waitingForOutput = false, true

		default:
			// JSON line.
			if waitingForInput {
				if inputLine != "" {
					return nil, fmt.Errorf("line %d: multiple input JSON lines for case %q", lineNum, currentName)
				}
				inputLine = line
			} else if waitingForOutput {
				if outputLine != "" {
					return nil, fmt.Errorf("line %d: multiple output JSON lines for case %q", lineNum, currentName)
				}
				outputLine = line
			} else {
				return nil, fmt.Errorf("line %d: unexpected JSON without header/@: %q", lineNum, line)
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scan %s: %w", path, err)
	}
	if err := flushCase(); err != nil {
		return nil, err
	}

	return cases, nil
}

// -----------------------------------------------------------------------------
// Typed cases + loader
// -----------------------------------------------------------------------------

// FindWellConnectedClustersCase is one fully-parsed test case.
type FindWellConnectedClustersCase struct {
	Name             string
	Graph            *Graph
	Subset           []int
	RefinedPartition []int
	Gamma            float64
	Candidates       []int
}

// loadFindWellConnectedClustersCases converts raw cases into typed cases,
// building a *Graph from the "directed" adjacency matrix for each case.
func loadFindWellConnectedClustersCases(path string) ([]FindWellConnectedClustersCase, error) {
	rawCases, err := parseFindWellConnectedClustersRawCases(path)
	if err != nil {
		return nil, err
	}

	// JSON shapes that match the test file.
	type input struct {
		Directed         [][]float64 `json:"directed"`
		Subset           []int       `json:"subset"`
		RefinedPartition []int       `json:"refinedPartition"`
		Gamma            float64     `json:"gamma"`
	}
	type output struct {
		Candidates []int `json:"candidates"`
	}

	cases := make([]FindWellConnectedClustersCase, 0, len(rawCases))
	for _, rc := range rawCases {
		var in input
		if err := json.Unmarshal([]byte(rc.InputLine), &in); err != nil {
			return nil, fmt.Errorf("case %q input: %w", rc.Name, err)
		}
		var out output
		if err := json.Unmarshal([]byte(rc.OutputLine), &out); err != nil {
			return nil, fmt.Errorf("case %q output: %w", rc.Name, err)
		}

		g := buildGraphFromDirected(in.Directed)

		cases = append(cases, FindWellConnectedClustersCase{
			Name:             rc.Name,
			Graph:            g,
			Subset:           in.Subset,
			RefinedPartition: in.RefinedPartition,
			Gamma:            in.Gamma,
			Candidates:       out.Candidates,
		})
	}

	return cases, nil
}

// -----------------------------------------------------------------------------
// Test function
// -----------------------------------------------------------------------------

func TestFindWellConnectedClusters_FromFile(t *testing.T) {

	const path = "clustering_tests/FindWellConnectedClusters.txt"

	cases, err := loadFindWellConnectedClustersCases(path)
	if err != nil {
		t.Fatalf("failed to load FindWellConnectedClusters cases: %v", err)
	}

	for i, tc := range cases {
		tc := tc // capture
		t.Run(tcName2(i, tc.Name), func(t *testing.T) {
			got := tc.Graph.FindWellConnectedClusters(tc.Subset, tc.RefinedPartition, tc.Gamma)
			if !equalIntSlicesIgnoringOrder(got, tc.Candidates) {
				t.Fatalf("candidates = %v, want %v", got, tc.Candidates)
			}
		})
	}
}

//parser for SampleCommunity

// SampleCommunityCase is one fully parsed test case
// for Graph.SampleCommunity (receiver g is unused in the function body).
type SampleCommunityCase struct {
	Name     string
	Clusters []int
	Probs    map[int]float64
	Selected int
}

// loadSampleCommunityCases reads a file with blocks:
//
//	# Case name
//	{"clusters":[...],"probs":{"0":1.0,...}}
//	@ some comment
//	{"selected":<int>}
//
// and returns all parsed cases.
func loadSampleCommunityCases(path string) ([]SampleCommunityCase, error) {
	rawCases, err := parseSampleCommunityRawCases(path)
	if err != nil {
		return nil, err
	}

	// JSON shapes for input/output.
	type input struct {
		Clusters []int           `json:"clusters"`
		Probs    map[int]float64 `json:"probs"`
	}
	type output struct {
		Selected int `json:"selected"`
	}

	cases := make([]SampleCommunityCase, 0, len(rawCases))
	for _, rc := range rawCases {
		var in input
		if err := json.Unmarshal([]byte(rc.InputLine), &in); err != nil {
			return nil, err
		}

		var out output
		if err := json.Unmarshal([]byte(rc.OutputLine), &out); err != nil {
			return nil, err
		}

		cases = append(cases, SampleCommunityCase{
			Name:     rc.Name,
			Clusters: in.Clusters,
			Probs:    in.Probs,
			Selected: out.Selected,
		})
	}

	return cases, nil
}

// ---------- internal raw parser for "# / JSON / @ / JSON" ----------

type sampleCommunityRawCase struct {
	Name       string
	InputLine  string
	OutputLine string
}

func parseSampleCommunityRawCases(path string) ([]sampleCommunityRawCase, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var (
		cases            []sampleCommunityRawCase
		currentName      string
		pendingInput     string
		waitingForInput  bool
		waitingForOutput bool
	)

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		switch {
		case strings.HasPrefix(line, "#"):
			// start a new case
			currentName = strings.TrimSpace(strings.TrimPrefix(line, "#"))
			pendingInput = ""
			waitingForInput = true
			waitingForOutput = false

		case strings.HasPrefix(line, "@"):
			// comment between input and output
			waitingForOutput = true

		default:
			if waitingForInput {
				// this is the input JSON line
				pendingInput = line
				waitingForInput = false
			} else if waitingForOutput {
				// this is the output JSON line → finalize case
				cases = append(cases, sampleCommunityRawCase{
					Name:       currentName,
					InputLine:  pendingInput,
					OutputLine: line,
				})
				waitingForOutput = false
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return cases, nil
}

func TestSampleCommunity_FromFile(t *testing.T) {
	const testFile = "clustering_tests/SampleCommunity.txt"

	cases, err := loadSampleCommunityCases(testFile)
	if err != nil {
		t.Fatalf("failed to parse %s: %v", testFile, err)
	}
	if len(cases) == 0 {
		t.Fatalf("no test cases in %s", testFile)
	}

	// receiver g is unused inside SampleCommunity, so nil is fine
	var g *Graph

	for idx, tc := range cases {
		tc := tc // capture range variable

		name := fmt.Sprintf("%03d_%s", idx, tc.Name)

		t.Run(name, func(t *testing.T) {
			got := g.SampleCommunity(tc.Clusters, tc.Probs)
			if got != tc.Selected {
				t.Fatalf("SampleCommunity(%v, %v) = %d, want %d",
					tc.Clusters, tc.Probs, got, tc.Selected)
			}
		})
	}
}

//pareser and test function for SampleCommunity

// parser for ComputeMoveProbabilities

type ComputeMoveProbabilityRawCase struct {
	Name       string
	InputLine  string
	OutputLine string
}

func parseComputeMoveProbabilityRawCases(path string) ([]ComputeMoveProbabilityRawCase, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)

	var (
		cases                             []ComputeMoveProbabilityRawCase
		currentName                       string
		inputLine, outputLine             string
		waitingForInput, waitingForOutput bool
		lineNum                           int
	)

	flushCase := func() error {
		if inputLine == "" && outputLine == "" {
			return nil
		}
		if inputLine == "" || outputLine == "" {
			return fmt.Errorf("incomplete case %q: missing input or output", currentName)
		}
		name := strings.TrimSpace(currentName)
		if name == "" {
			name = fmt.Sprintf("unnamed_%d", len(cases)+1)
		}
		cases = append(cases, ComputeMoveProbabilityRawCase{
			Name:       name,
			InputLine:  inputLine,
			OutputLine: outputLine,
		})
		currentName = ""
		inputLine, outputLine = "", ""
		waitingForInput, waitingForOutput = false, false
		return nil
	}

	for scanner.Scan() {
		lineNum++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		switch {
		case strings.HasPrefix(line, "#"):
			// New case header – flush previous case (if any).
			if err := flushCase(); err != nil {
				return nil, fmt.Errorf("line %d: %w", lineNum, err)
			}
			currentName = strings.TrimSpace(strings.TrimPrefix(line, "#"))
			waitingForInput, waitingForOutput = true, false

		case strings.HasPrefix(line, "@"):
			// Comment line – next JSON is the output.
			waitingForInput, waitingForOutput = false, true

		default:
			// JSON line.
			if waitingForInput {
				if inputLine != "" {
					return nil, fmt.Errorf("line %d: multiple input JSON lines for case %q", lineNum, currentName)
				}
				inputLine = line
			} else if waitingForOutput {
				if outputLine != "" {
					return nil, fmt.Errorf("line %d: multiple output JSON lines for case %q", lineNum, currentName)
				}
				outputLine = line
			} else {
				return nil, fmt.Errorf("line %d: unexpected JSON without header/@: %q", lineNum, line)
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scan %s: %w", path, err)
	}
	if err := flushCase(); err != nil {
		return nil, err
	}

	return cases, nil
}

// ---------- typed cases + loader ----------

// ComputeMoveProbabilityCase is one fully parsed test case.
type ComputeMoveProbabilityCase struct {
	Name              string
	CurrNode          int
	CandidateClusters []int
	RefinedPartition  []int
	Theta             float64
	Resolution        float64
	Probs             map[int]float64
}

// loadComputeMoveProbabilityCases reads a file with blocks:
//
//	# Case name
//	{"currNode":...,"candidateClusters":[...],"refinedPartition":[...],"theta":...,"resolution":...}
//	@ some comment
//	{"probs":{"1":1.0,...}}
//
// and returns all parsed cases.
func loadComputeMoveProbabilityCases(path string) ([]ComputeMoveProbabilityCase, error) {
	rawCases, err := parseComputeMoveProbabilityRawCases(path)
	if err != nil {
		return nil, err
	}

	// JSON shapes for input/output.
	type input struct {
		CurrNode          int     `json:"currNode"`
		CandidateClusters []int   `json:"candidateClusters"`
		RefinedPartition  []int   `json:"refinedPartition"`
		Theta             float64 `json:"theta"`
		Resolution        float64 `json:"resolution"`
	}
	type output struct {
		Probs map[int]float64 `json:"probs"`
	}

	cases := make([]ComputeMoveProbabilityCase, 0, len(rawCases))
	for _, rc := range rawCases {
		var in input
		if err := json.Unmarshal([]byte(rc.InputLine), &in); err != nil {
			return nil, fmt.Errorf("case %q input: %w", rc.Name, err)
		}
		var out output
		if err := json.Unmarshal([]byte(rc.OutputLine), &out); err != nil {
			return nil, fmt.Errorf("case %q output: %w", rc.Name, err)
		}

		cases = append(cases, ComputeMoveProbabilityCase{
			Name:              rc.Name,
			CurrNode:          in.CurrNode,
			CandidateClusters: in.CandidateClusters,
			RefinedPartition:  in.RefinedPartition,
			Theta:             in.Theta,
			Resolution:        in.Resolution,
			Probs:             out.Probs,
		})
	}

	return cases, nil
}

// ---------- test for Graph.ComputeMoveProbability ----------

func TestComputeMoveProbability_FromFile(t *testing.T) {
	const path = "clustering_tests/ComputeMoveProbability.txt"

	cases, err := loadComputeMoveProbabilityCases(path)
	if err != nil {
		t.Fatalf("failed to load test cases from %s: %v", path, err)
	}

	// TODO: if ModularityGain depends on graph structure, build an appropriate
	// test graph here instead of using the zero value.
	var g Graph

	for i, tc := range cases {
		tc := tc // capture
		t.Run(tcName2(i, tc.Name), func(t *testing.T) {
			// Build nodesByCluster map from the refined partition
			nodesByCluster := g.NodesByCluster(tc.RefinedPartition)
			got := g.ComputeMoveProbability(
				tc.CurrNode,
				tc.CandidateClusters,
				tc.RefinedPartition,
				tc.Theta,
				tc.Resolution,
				nodesByCluster,
			)

			// Check same number of entries.
			if len(got) != len(tc.Probs) {
				t.Fatalf("len(got)=%d, want %d; got=%v, want=%v",
					len(got), len(tc.Probs), got, tc.Probs)
			}

			// Compare each probability with a small tolerance.
			const tol = 1e-9
			for cid, want := range tc.Probs {
				seen, ok := got[cid]
				if !ok {
					t.Fatalf("missing probability for cluster %d in got=%v", cid, got)
				}
				if !floatAlmostEqual2(seen, want, tol) {
					t.Fatalf("probs[%d]=%v, want %v (case %q)",
						cid, seen, want, tc.Name)
				}
			}
		})
	}
}

// test function for ComputeMoveProbability

// FindWellConnectedNodesCase is one fully parsed test case
// for Graph.FindWellConnectedNodes.
type FindWellConnectedNodesCase struct {
	Name      string
	Graph     *Graph
	Subset    []int
	Partition []int
	Gamma     float64
	Connected []int
}

// loadFindWellConnectedNodesCases reads a file with blocks:
//
//	# Case name
//	{"graphNodes":...,"edges":{...},"totalWeight":...,
//	 "subset":[...],"partition":[...],"gamma":...}
//	@ some comment
//	{"connected":[...]}
//
// and returns all parsed cases.
func loadFindWellConnectedNodesCases(path string) ([]FindWellConnectedNodesCase, error) {
	rawCases, err := parseFindWellConnectedNodesRawCases(path)
	if err != nil {
		return nil, err
	}

	// JSON shapes for input / output.
	type input struct {
		GraphNodes  int                `json:"graphNodes"`
		Edges       map[int][]EdgeJSON `json:"edges"`
		TotalWeight float64            `json:"totalWeight"`
		Subset      []int              `json:"subset"`
		Partition   []int              `json:"partition"`
		Gamma       float64            `json:"gamma"`
	}
	type output struct {
		Connected []int `json:"connected"`
	}

	cases := make([]FindWellConnectedNodesCase, 0, len(rawCases))
	for _, rc := range rawCases {
		var in input
		if err := json.Unmarshal([]byte(rc.InputLine), &in); err != nil {
			return nil, err
		}

		var out output
		if err := json.Unmarshal([]byte(rc.OutputLine), &out); err != nil {
			return nil, err
		}

		// convert EdgeJSON → Edge to build Graph
		edges := make(map[int][]Edge, len(in.Edges))
		for from, ejs := range in.Edges {
			es := make([]Edge, len(ejs))
			for i, ej := range ejs {
				es[i] = Edge{
					To:     ej.To,
					Weight: ej.Weight,
				}
			}
			edges[from] = es
		}

		g := &Graph{
			Nodes:       in.GraphNodes,
			Edges:       edges,
			TotalWeight: in.TotalWeight,
		}

		cases = append(cases, FindWellConnectedNodesCase{
			Name:      rc.Name,
			Graph:     g,
			Subset:    in.Subset,
			Partition: in.Partition,
			Gamma:     in.Gamma,
			Connected: out.Connected,
		})
	}

	return cases, nil
}

// ---------- internal raw parser for "# / JSON / @ / JSON" ----------

type findWellConnectedNodesRawCase struct {
	Name       string
	InputLine  string
	OutputLine string
}

func parseFindWellConnectedNodesRawCases(path string) ([]findWellConnectedNodesRawCase, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var (
		cases            []findWellConnectedNodesRawCase
		currentName      string
		pendingInput     string
		waitingForInput  bool
		waitingForOutput bool
	)

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		switch {
		case strings.HasPrefix(line, "#"):
			// start new case
			currentName = strings.TrimSpace(strings.TrimPrefix(line, "#"))
			pendingInput = ""
			waitingForInput = true
			waitingForOutput = false

		case strings.HasPrefix(line, "@"):
			// comment line between input and output
			waitingForOutput = true

		default:
			if waitingForInput {
				// this is the input JSON line
				pendingInput = line
				waitingForInput = false
			} else if waitingForOutput {
				// this is the output JSON line → finalize case
				cases = append(cases, findWellConnectedNodesRawCase{
					Name:       currentName,
					InputLine:  pendingInput,
					OutputLine: line,
				})
				waitingForOutput = false
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return cases, nil
}

// parser for EdgestoCluster

// EdgesToClusterCase is one fully parsed test case for Graph.EdgesToCluster.
type EdgesToClusterCase struct {
	Name    string
	Graph   *Graph
	Node    int
	Cluster []int
	Sum     float64
}

// loadEdgesToClusterCases reads a file with blocks:
//
//	# Case name
//	{"nodes":...,"edges":{...},"totalWeight":...,"node":<int>,"cluster":[...]}
//	@ some comment
//	{"sum":<float>}
//
// and returns all parsed cases.
func loadEdgesToClusterCases(path string) ([]EdgesToClusterCase, error) {
	rawCases, err := parseEdgesToClusterRawCases(path)
	if err != nil {
		return nil, err
	}

	// JSON shapes for input/output.
	type input struct {
		Nodes       int                `json:"nodes"`
		Edges       map[int][]EdgeJSON `json:"edges"`
		TotalWeight float64            `json:"totalWeight"`
		Node        int                `json:"node"`
		Cluster     []int              `json:"cluster"`
	}
	type output struct {
		Sum float64 `json:"sum"`
	}

	cases := make([]EdgesToClusterCase, 0, len(rawCases))
	for _, rc := range rawCases {
		var in input
		if err := json.Unmarshal([]byte(rc.InputLine), &in); err != nil {
			return nil, err
		}

		var out output
		if err := json.Unmarshal([]byte(rc.OutputLine), &out); err != nil {
			return nil, err
		}

		// convert EdgeJSON → Edge to build Graph
		edges := make(map[int][]Edge, len(in.Edges))
		for from, ejs := range in.Edges {
			es := make([]Edge, len(ejs))
			for i, ej := range ejs {
				es[i] = Edge{
					To:     ej.To,
					Weight: ej.Weight,
				}
			}
			edges[from] = es
		}

		g := &Graph{
			Nodes:       in.Nodes,
			Edges:       edges,
			TotalWeight: in.TotalWeight,
		}

		cases = append(cases, EdgesToClusterCase{
			Name:    rc.Name,
			Graph:   g,
			Node:    in.Node,
			Cluster: in.Cluster,
			Sum:     out.Sum,
		})
	}

	return cases, nil
}

// ---------- internal raw parser for "# / JSON / @ / JSON" ----------

type edgesToClusterRawCase struct {
	Name       string
	InputLine  string
	OutputLine string
}

func parseEdgesToClusterRawCases(path string) ([]edgesToClusterRawCase, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var (
		cases            []edgesToClusterRawCase
		currentName      string
		pendingInput     string
		waitingForInput  bool
		waitingForOutput bool
	)

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		switch {
		case strings.HasPrefix(line, "#"):
			// start a new case
			currentName = strings.TrimSpace(strings.TrimPrefix(line, "#"))
			pendingInput = ""
			waitingForInput = true
			waitingForOutput = false

		case strings.HasPrefix(line, "@"):
			// comment between input and output
			waitingForOutput = true

		default:
			if waitingForInput {
				// this is the input JSON line
				pendingInput = line
				waitingForInput = false
			} else if waitingForOutput {
				// this is the output JSON line → finalize case
				cases = append(cases, edgesToClusterRawCase{
					Name:       currentName,
					InputLine:  pendingInput,
					OutputLine: line,
				})
				waitingForOutput = false
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return cases, nil
}

// test function for EdgestoCluster
func TestFindWellConnectedNodes_FromFile(t *testing.T) {
	const path = "clustering_tests/FindWellConnectedNodes.txt" // adjust if you use a subdir

	cases, err := loadFindWellConnectedNodesCases(path)
	if err != nil {
		t.Fatalf("failed to load FindWellConnectedNodes cases: %v", err)
	}

	for i, tc := range cases {
		tc := tc // capture
		t.Run(tcName2(i, tc.Name), func(t *testing.T) {
			got := tc.Graph.FindWellConnectedNodes(tc.Subset, tc.Partition, tc.Gamma)
			if !equalIntSlicesIgnoringOrder(got, tc.Connected) {
				t.Fatalf("connected = %v, want %v", got, tc.Connected)
			}
		})
	}
}

// parser for ClusterDegree

// ClusterDegreeCase is one fully parsed test case for Graph.ClusterDegree.
type ClusterDegreeCase struct {
	Name    string
	Graph   *Graph
	Cluster []int
	Degree  float64
}

// loadClusterDegreeCases reads a file with blocks:
//
//	# Case name
//	{"nodes":...,"edges":{...},"totalWeight":...,"cluster":[...]}
//	@ some comment
//	{"degree":<float>}
//
// and returns all parsed cases.
func loadClusterDegreeCases(path string) ([]ClusterDegreeCase, error) {
	rawCases, err := parseClusterDegreeRawCases(path)
	if err != nil {
		return nil, err
	}

	// JSON shapes for input / output.
	type input struct {
		Nodes       int                `json:"nodes"`
		Edges       map[int][]EdgeJSON `json:"edges"`
		TotalWeight float64            `json:"totalWeight"`
		Cluster     []int              `json:"cluster"`
	}
	type output struct {
		Degree float64 `json:"degree"`
	}

	cases := make([]ClusterDegreeCase, 0, len(rawCases))
	for _, rc := range rawCases {
		var in input
		if err := json.Unmarshal([]byte(rc.InputLine), &in); err != nil {
			return nil, err
		}

		var out output
		if err := json.Unmarshal([]byte(rc.OutputLine), &out); err != nil {
			return nil, err
		}

		// convert EdgeJSON → Edge to build Graph
		edges := make(map[int][]Edge, len(in.Edges))
		for from, ejs := range in.Edges {
			es := make([]Edge, len(ejs))
			for i, ej := range ejs {
				es[i] = Edge{
					To:     ej.To,
					Weight: ej.Weight,
				}
			}
			edges[from] = es
		}

		g := &Graph{
			Nodes:       in.Nodes,
			Edges:       edges,
			TotalWeight: in.TotalWeight,
		}

		cases = append(cases, ClusterDegreeCase{
			Name:    rc.Name,
			Graph:   g,
			Cluster: in.Cluster,
			Degree:  out.Degree,
		})
	}

	return cases, nil
}

// ---------- internal raw parser for "# / JSON / @ / JSON" ----------

type clusterDegreeRawCase struct {
	Name       string
	InputLine  string
	OutputLine string
}

func parseClusterDegreeRawCases(path string) ([]clusterDegreeRawCase, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var (
		cases            []clusterDegreeRawCase
		currentName      string
		pendingInput     string
		waitingForInput  bool
		waitingForOutput bool
	)

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		switch {
		case strings.HasPrefix(line, "#"):
			// start a new case
			currentName = strings.TrimSpace(strings.TrimPrefix(line, "#"))
			pendingInput = ""
			waitingForInput = true
			waitingForOutput = false

		case strings.HasPrefix(line, "@"):
			// comment between input and output
			waitingForOutput = true

		default:
			if waitingForInput {
				// this is the input JSON line
				pendingInput = line
				waitingForInput = false
			} else if waitingForOutput {
				// this is the output JSON line → finalize case
				cases = append(cases, clusterDegreeRawCase{
					Name:       currentName,
					InputLine:  pendingInput,
					OutputLine: line,
				})
				waitingForOutput = false
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return cases, nil
}

// ---------- test function for Graph.FindWellConnectedNodes ----------

func TestClusterDegree_FromFile(t *testing.T) {
	const path = "clustering_tests/ClusterDegree.txt" // adjust path if your file lives elsewhere

	cases, err := loadClusterDegreeCases(path)
	if err != nil {
		t.Fatalf("failed to load ClusterDegree cases: %v", err)
	}

	for i, tc := range cases {
		tc := tc // capture
		t.Run(tcName2(i, tc.Name), func(t *testing.T) {
			got := tc.Graph.ClusterDegree(tc.Cluster)

			// The expected values in ClusterDegree_tests.txt are simple (0, 3.5, 2.0, ...)
			// so exact comparison is fine. If you prefer tolerance, replace with your floatEquals.
			if got != tc.Degree {
				t.Fatalf("ClusterDegree(%v) = %v, want %v", tc.Cluster, got, tc.Degree)
			}
		})
	}
}

//parser for SingletonPartition

// SingletonCase is one fully parsed test case for Singleton.
type SingletonCase struct {
	Name      string
	Node      int
	Partition []int
	Singleton bool
}

// loadSingletonCases reads a file with blocks:
//
//	# Case name
//	{"node":<int>,"partition":[...]}
//	@ some comment
//	{"singleton":<bool>}
//
// and returns all parsed cases.
func loadSingletonCases(path string) ([]SingletonCase, error) {
	rawCases, err := parseSingletonRawCases(path)
	if err != nil {
		return nil, err
	}

	// JSON shapes for input/output.
	type input struct {
		Node      int   `json:"node"`
		Partition []int `json:"partition"`
	}
	type output struct {
		Singleton bool `json:"singleton"`
	}

	cases := make([]SingletonCase, 0, len(rawCases))
	for _, rc := range rawCases {
		var in input
		if err := json.Unmarshal([]byte(rc.InputLine), &in); err != nil {
			return nil, err
		}

		var out output
		if err := json.Unmarshal([]byte(rc.OutputLine), &out); err != nil {
			return nil, err
		}

		cases = append(cases, SingletonCase{
			Name:      rc.Name,
			Node:      in.Node,
			Partition: in.Partition,
			Singleton: out.Singleton,
		})
	}

	return cases, nil
}

// ---------- internal raw parser for "# / JSON / @ / JSON" ----------

type singletonRawCase struct {
	Name       string
	InputLine  string
	OutputLine string
}

func parseSingletonRawCases(path string) ([]singletonRawCase, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var (
		cases            []singletonRawCase
		currentName      string
		pendingInput     string
		waitingForInput  bool
		waitingForOutput bool
	)

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		switch {
		case strings.HasPrefix(line, "#"):
			// start new case
			currentName = strings.TrimSpace(strings.TrimPrefix(line, "#"))
			pendingInput = ""
			waitingForInput = true
			waitingForOutput = false

		case strings.HasPrefix(line, "@"):
			// comment between input and output
			waitingForOutput = true

		default:
			if waitingForInput {
				// input JSON line
				pendingInput = line
				waitingForInput = false
			} else if waitingForOutput {
				// output JSON line → finalize case
				cases = append(cases, singletonRawCase{
					Name:       currentName,
					InputLine:  pendingInput,
					OutputLine: line,
				})
				waitingForOutput = false
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return cases, nil
}

//test function for Singleton

func TestSingleton_FromFile(t *testing.T) {
	const path = "clustering_tests/Singleton.txt" // adjust if you use a subdir

	cases, err := loadSingletonCases(path)
	if err != nil {
		t.Fatalf("failed to load Singleton cases: %v", err)
	}

	for i, tc := range cases {
		tc := tc // capture
		t.Run(tcName2(i, tc.Name), func(t *testing.T) {
			got := Singleton(tc.Node, tc.Partition)
			if got != tc.Singleton {
				t.Fatalf("Singleton(%d, %v) = %v, want %v",
					tc.Node, tc.Partition, got, tc.Singleton)
			}
		})
	}
}

//parser for MoveNodes

// MoveNodesCase represents one test case for (*Graph).MoveNodes.
type MoveNodesCase struct {
	Name       string
	Graph      *Graph
	Partition  []int
	Resolution float64
	Expected   []int
}

// loadMoveNodesCases reads a file with blocks:
//
//	# Case name
//	{"nodes":...,"edges":{...},"totalWeight":...,"partition":[...],"resolution":...}
//	@ some comment
//	{"partition":[...]}   // expected final partition
//
// and returns all parsed cases.
func loadMoveNodesCases(path string) ([]MoveNodesCase, error) {
	rawCases, err := parseMoveNodesRawCases(path)
	if err != nil {
		return nil, err
	}

	// JSON shapes for input / output.
	type input struct {
		Nodes       int                `json:"nodes"`
		Edges       map[int][]EdgeJSON `json:"edges"`
		TotalWeight float64            `json:"totalWeight"`
		Partition   []int              `json:"partition"`
		Resolution  float64            `json:"resolution"`
	}
	type output struct {
		Partition []int `json:"partition"`
	}

	cases := make([]MoveNodesCase, 0, len(rawCases))
	for _, rc := range rawCases {
		var in input
		if err := json.Unmarshal([]byte(rc.InputLine), &in); err != nil {
			return nil, err
		}

		var out output
		if err := json.Unmarshal([]byte(rc.OutputLine), &out); err != nil {
			return nil, err
		}

		// convert EdgeJSON → Edge to build Graph
		edges := make(map[int][]Edge, len(in.Edges))
		for from, ejs := range in.Edges {
			es := make([]Edge, len(ejs))
			for i, ej := range ejs {
				es[i] = Edge{
					To:     ej.To,
					Weight: ej.Weight,
				}
			}
			edges[from] = es
		}

		g := &Graph{
			Nodes:       in.Nodes,
			Edges:       edges,
			TotalWeight: in.TotalWeight,
		}

		cases = append(cases, MoveNodesCase{
			Name:       rc.Name,
			Graph:      g,
			Partition:  in.Partition,
			Resolution: in.Resolution,
			Expected:   out.Partition,
		})
	}

	return cases, nil
}

// ---------- internal raw parser for "# / JSON / @ / JSON" ----------

type moveNodesRawCase struct {
	Name       string
	InputLine  string
	OutputLine string
}

func parseMoveNodesRawCases(path string) ([]moveNodesRawCase, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var (
		cases            []moveNodesRawCase
		currentName      string
		pendingInput     string
		waitingForInput  bool
		waitingForOutput bool
	)

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		switch {
		case strings.HasPrefix(line, "#"):
			// start a new case
			currentName = strings.TrimSpace(strings.TrimPrefix(line, "#"))
			pendingInput = ""
			waitingForInput = true
			waitingForOutput = false

		case strings.HasPrefix(line, "@"):
			// comment between input and output
			waitingForOutput = true

		default:
			if waitingForInput {
				// input JSON line
				pendingInput = line
				waitingForInput = false
			} else if waitingForOutput {
				// output JSON line → finalize case
				cases = append(cases, moveNodesRawCase{
					Name:       currentName,
					InputLine:  pendingInput,
					OutputLine: line,
				})
				waitingForOutput = false
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return cases, nil
}

func TestMoveNodes_FromFile(t *testing.T) {
	const path = "clustering_tests/MoveNodes.txt"

	cases, err := loadMoveNodesCases(path)
	if err != nil {
		t.Fatalf("failed to load MoveNodes cases: %v", err)
	}

	for i, tc := range cases {
		tc := tc // capture
		t.Run(tcName2(i, tc.Name), func(t *testing.T) {
			// copy so we don't mutate the test case struct's slice
			start := append([]int(nil), tc.Partition...)

			got := tc.Graph.MoveNodes(start, tc.Resolution, 10)

			if !equalIntSlices(got, tc.Partition) {
				t.Fatalf("MoveNodes(%v, %.4f) = %v, want %v",
					tc.Partition, tc.Resolution, got, tc.Partition)
			}
		})
	}
}

// equalIntSlices checks equality with order (unlike the ignoring-order helper).
func equalIntSlices(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

//parser for FindBestClustering

// FindBestClusterCase represents one test case for FindBestCluster.
type FindBestClusterCase struct {
	Name              string
	Graph             *Graph
	Node              int
	CandidateClusters []int
	Partition         []int
	Resolution        float64
	BestCluster       int
	Improved          bool
}

// loadFindBestClusterCases reads a file with blocks:
//
//	# Case name
//	{"nodes":...,"edges":{...},"totalWeight":...,
//	 "node":<int>,"candidateClusters":[...],
//	 "partition":[...],"resolution":...}
//	@ some comment
//	{"bestCluster":<int>,"improved":<bool>}
//
// and returns all parsed cases.
func loadFindBestClusterCases(path string) ([]FindBestClusterCase, error) {
	rawCases, err := parseFindBestClusterRawCases(path)
	if err != nil {
		return nil, err
	}

	// JSON input/output shapes.
	type input struct {
		Nodes             int                `json:"nodes"`
		Edges             map[int][]EdgeJSON `json:"edges"`
		TotalWeight       float64            `json:"totalWeight"`
		Node              int                `json:"node"`
		CandidateClusters []int              `json:"candidateClusters"`
		Partition         []int              `json:"partition"`
		Resolution        float64            `json:"resolution"`
	}
	type output struct {
		BestCluster int  `json:"bestCluster"`
		Improved    bool `json:"improved"`
	}

	cases := make([]FindBestClusterCase, 0, len(rawCases))
	for _, rc := range rawCases {
		var in input
		if err := json.Unmarshal([]byte(rc.InputLine), &in); err != nil {
			return nil, err
		}

		var out output
		if err := json.Unmarshal([]byte(rc.OutputLine), &out); err != nil {
			return nil, err
		}

		// convert EdgeJSON → Edge to build Graph
		edges := make(map[int][]Edge, len(in.Edges))
		for from, ejs := range in.Edges {
			es := make([]Edge, len(ejs))
			for i, ej := range ejs {
				es[i] = Edge{
					To:     ej.To,
					Weight: ej.Weight,
				}
			}
			edges[from] = es
		}

		g := &Graph{
			Nodes:       in.Nodes,
			Edges:       edges,
			TotalWeight: in.TotalWeight,
		}

		cases = append(cases, FindBestClusterCase{
			Name:              rc.Name,
			Graph:             g,
			Node:              in.Node,
			CandidateClusters: in.CandidateClusters,
			Partition:         in.Partition,
			Resolution:        in.Resolution,
			BestCluster:       out.BestCluster,
			Improved:          out.Improved,
		})
	}

	return cases, nil
}

// ---------- internal raw parser for "# / JSON / @ / JSON" ----------

type findBestClusterRawCase struct {
	Name       string
	InputLine  string
	OutputLine string
}

func parseFindBestClusterRawCases(path string) ([]findBestClusterRawCase, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var (
		cases            []findBestClusterRawCase
		currentName      string
		pendingInput     string
		waitingForInput  bool
		waitingForOutput bool
	)

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		switch {
		case strings.HasPrefix(line, "#"):
			// start a new case
			currentName = strings.TrimSpace(strings.TrimPrefix(line, "#"))
			pendingInput = ""
			waitingForInput = true
			waitingForOutput = false

		case strings.HasPrefix(line, "@"):
			// comment line between input and output
			waitingForOutput = true

		default:
			if waitingForInput {
				// this is the input JSON line
				pendingInput = line
				waitingForInput = false
			} else if waitingForOutput {
				// this is the output JSON line → finalize case
				cases = append(cases, findBestClusterRawCase{
					Name:       currentName,
					InputLine:  pendingInput,
					OutputLine: line,
				})
				waitingForOutput = false
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return cases, nil
}

func TestFindBestCluster_FromFile(t *testing.T) {
	const path = "clustering_tests/FindBestCluster.txt" // adjust if it's in a subdir

	cases, err := loadFindBestClusterCases(path)
	if err != nil {
		t.Fatalf("failed to load FindBestCluster cases: %v", err)
	}

	for i, tc := range cases {
		tc := tc // capture range variable
		t.Run(tcName2(i, tc.Name), func(t *testing.T) {
			gotCluster, gotImproved := FindBestCluster(
				tc.Node,
				tc.Graph,
				tc.CandidateClusters,
				tc.Partition,
				tc.Resolution,
			)

			if gotCluster != tc.BestCluster || gotImproved != tc.Improved {
				t.Fatalf(
					"FindBestCluster(node=%d, cand=%v, part=%v, res=%.4f) = (%d, %v); want (%d, %v)",
					tc.Node, tc.CandidateClusters, tc.Partition, tc.Resolution,
					gotCluster, gotImproved,
					tc.BestCluster, tc.Improved,
				)
			}
		})
	}
}

//parser for RandomNodeOrder

// RandomNodeOrderCase represents one test case for RandomNodeOrder.
type RandomNodeOrderCase struct {
	Name        string
	N           int
	Seed        int64
	ExpectedLen int
}

// loadRandomNodeOrderCases reads a file with blocks:
//
//	# Case name
//	{"n":<int>,"seed":<int>}
//	@ some comment
//	{"expectedLen":<int>}
//
// and returns all parsed cases.
func loadRandomNodeOrderCases(path string) ([]RandomNodeOrderCase, error) {
	rawCases, err := parseRandomNodeOrderRawCases(path)
	if err != nil {
		return nil, err
	}

	// JSON shapes for input/output.
	type input struct {
		N    int   `json:"n"`
		Seed int64 `json:"seed"`
	}
	type output struct {
		ExpectedLen int `json:"expectedLen"`
	}

	cases := make([]RandomNodeOrderCase, 0, len(rawCases))
	for _, rc := range rawCases {
		var in input
		if err := json.Unmarshal([]byte(rc.InputLine), &in); err != nil {
			return nil, err
		}

		var out output
		if err := json.Unmarshal([]byte(rc.OutputLine), &out); err != nil {
			return nil, err
		}

		cases = append(cases, RandomNodeOrderCase{
			Name:        rc.Name,
			N:           in.N,
			Seed:        in.Seed,
			ExpectedLen: out.ExpectedLen,
		})
	}

	return cases, nil
}

// ---------- internal raw parser for "# / JSON / @ / JSON" ----------

type randomNodeOrderRawCase struct {
	Name       string
	InputLine  string
	OutputLine string
}

func parseRandomNodeOrderRawCases(path string) ([]randomNodeOrderRawCase, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var (
		cases            []randomNodeOrderRawCase
		currentName      string
		pendingInput     string
		waitingForInput  bool
		waitingForOutput bool
	)

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		switch {
		case strings.HasPrefix(line, "#"):
			// start a new case
			currentName = strings.TrimSpace(strings.TrimPrefix(line, "#"))
			pendingInput = ""
			waitingForInput = true
			waitingForOutput = false

		case strings.HasPrefix(line, "@"):
			// comment between input and output
			waitingForOutput = true

		default:
			if waitingForInput {
				// this is the input JSON line
				pendingInput = line
				waitingForInput = false
			} else if waitingForOutput {
				// this is the output JSON line → finalize case
				cases = append(cases, randomNodeOrderRawCase{
					Name:       currentName,
					InputLine:  pendingInput,
					OutputLine: line,
				})
				waitingForOutput = false
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return cases, nil
}

func TestRandomNodeOrder_FromFile(t *testing.T) {
	const path = "clustering_tests/RandomNodeOrder.txt"

	cases, err := loadRandomNodeOrderCases(path)
	if err != nil {
		t.Fatalf("failed to load RandomNodeOrder cases: %v", err)
	}

	for i, tc := range cases {
		tc := tc // capture
		t.Run(tcName2(i, tc.Name), func(t *testing.T) {
			// Use the provided seed so tests are reproducible.
			rand.Seed(tc.Seed)

			got := RandomNodeOrder(tc.N)

			// 1) Check length.
			if len(got) != tc.ExpectedLen {
				t.Fatalf("len(RandomNodeOrder(%d)) = %d, want %d",
					tc.N, len(got), tc.ExpectedLen)
			}

			// 2) For n > 0, ensure it's a permutation of [0..n-1].
			if tc.N > 0 {
				if !isPermutation0toNminus1(got, tc.N) {
					t.Fatalf("RandomNodeOrder(%d) = %v is not a permutation of [0..%d]",
						tc.N, got, tc.N-1)
				}
			}
		})
	}
}

// isPermutation0toNminus1 checks that s is a permutation of [0,1,...,n-1].
func isPermutation0toNminus1(s []int, n int) bool {
	if len(s) != n {
		return false
	}
	seen := make([]bool, n)
	for _, v := range s {
		if v < 0 || v >= n {
			return false
		}
		if seen[v] {
			return false
		}
		seen[v] = true
	}
	for _, ok := range seen {
		if !ok {
			return false
		}
	}
	return true
}

// parser for FindCandidateClusters

// FindCandidateClustersCase is one test case for FindCandidateClusters.
type FindCandidateClustersCase struct {
	Name       string
	Node       int
	Edges      []Edge
	Partition  []int
	Candidates []int
}

// loadFindCandidateClustersCases reads a file with blocks:
//
//	# Case name
//	{"node":...,"edges":[{"to":...,"weight":...},...],"partition":[...]}
//	@ some comment
//	{"candidates":[...]}
//
// and returns all parsed cases.
func loadFindCandidateClustersCases(path string) ([]FindCandidateClustersCase, error) {
	rawCases, err := parseFindCandidateClustersRawCases(path)
	if err != nil {
		return nil, err
	}

	// JSON shapes for input/output.
	type input struct {
		Node      int        `json:"node"`
		Edges     []EdgeJSON `json:"edges"`
		Partition []int      `json:"partition"`
	}
	type output struct {
		Candidates []int `json:"candidates"`
	}

	cases := make([]FindCandidateClustersCase, 0, len(rawCases))
	for _, rc := range rawCases {
		var in input
		if err := json.Unmarshal([]byte(rc.InputLine), &in); err != nil {
			return nil, err
		}

		var out output
		if err := json.Unmarshal([]byte(rc.OutputLine), &out); err != nil {
			return nil, err
		}

		// convert EdgeJSON → Edge
		edges := make([]Edge, len(in.Edges))
		for i, ej := range in.Edges {
			edges[i] = Edge{
				To:     ej.To,
				Weight: ej.Weight,
			}
		}

		cases = append(cases, FindCandidateClustersCase{
			Name:       rc.Name,
			Node:       in.Node,
			Edges:      edges,
			Partition:  in.Partition,
			Candidates: out.Candidates,
		})
	}

	return cases, nil
}

// ---------- internal raw parser for "# / JSON / @ / JSON" ----------

type findCandidateClustersRawCase struct {
	Name       string
	InputLine  string
	OutputLine string
}

func parseFindCandidateClustersRawCases(path string) ([]findCandidateClustersRawCase, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var (
		cases            []findCandidateClustersRawCase
		currentName      string
		pendingInput     string
		waitingForInput  bool
		waitingForOutput bool
	)

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		switch {
		case strings.HasPrefix(line, "#"):
			// start a new case
			currentName = strings.TrimSpace(strings.TrimPrefix(line, "#"))
			pendingInput = ""
			waitingForInput = true
			waitingForOutput = false

		case strings.HasPrefix(line, "@"):
			// comment between input and output
			waitingForOutput = true

		default:
			if waitingForInput {
				// this is the input JSON line
				pendingInput = line
				waitingForInput = false
			} else if waitingForOutput {
				// this is the output JSON line → finalize case
				cases = append(cases, findCandidateClustersRawCase{
					Name:       currentName,
					InputLine:  pendingInput,
					OutputLine: line,
				})
				waitingForOutput = false
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return cases, nil
}

func TestFindCandidateClusters_FromFile(t *testing.T) {
	const path = "clustering_tests/FindCandidateCluster.txt" // adjust if you use a subdir

	cases, err := loadFindCandidateClustersCases(path)
	if err != nil {
		t.Fatalf("failed to load FindCandidateClusters cases: %v", err)
	}

	for i, tc := range cases {
		tc := tc // capture range variable
		t.Run(tcName2(i, tc.Name), func(t *testing.T) {
			got := FindCandidateClusters(tc.Node, tc.Edges, tc.Partition)

			// order & duplicates matter (see your Case 5), so we use exact slice equality
			if !equalIntSlices(got, tc.Candidates) {
				t.Fatalf("FindCandidateClusters(node=%d, edges=%v, partition=%v) = %v, want %v",
					tc.Node, tc.Edges, tc.Partition, got, tc.Candidates)
			}
		})
	}
}

//parser for ModularityGain

// ModularityGainCase is one test case for Graph.ModularityGain.
type ModularityGainCase struct {
	Name       string
	Graph      *Graph
	I          int
	Cluster    int
	Partition  []int
	Resolution float64
	DeltaQ     float64
}

// loadModularityGainCases reads:
//
//	# Case name
//	{"nodes":...,"edges":{...},"totalWeight":...,"i":...,"cluster":...,
//	 "partition":[...],"resolution":...}
//	@ comment
//	{"deltaQ":...}
func loadModularityGainCases(path string) ([]ModularityGainCase, error) {
	rawCases, err := parseModularityGainRawCases(path)
	if err != nil {
		return nil, err
	}

	type input struct {
		Nodes       int                `json:"nodes"`
		Edges       map[int][]EdgeJSON `json:"edges"`
		TotalWeight float64            `json:"totalWeight"`
		I           int                `json:"i"`
		Cluster     int                `json:"cluster"`
		Partition   []int              `json:"partition"`
		Resolution  float64            `json:"resolution"`
	}
	type output struct {
		DeltaQ float64 `json:"deltaQ"`
	}

	var cases []ModularityGainCase
	for _, rc := range rawCases {
		var in input
		if err := json.Unmarshal([]byte(rc.InputLine), &in); err != nil {
			return nil, err
		}

		var out output
		if err := json.Unmarshal([]byte(rc.OutputLine), &out); err != nil {
			return nil, err
		}

		edges := make(map[int][]Edge, len(in.Edges))
		for from, ejs := range in.Edges {
			es := make([]Edge, len(ejs))
			for i, ej := range ejs {
				es[i] = Edge{To: ej.To, Weight: ej.Weight}
			}
			edges[from] = es
		}

		g := &Graph{
			Nodes:       in.Nodes,
			Edges:       edges,
			TotalWeight: in.TotalWeight,
		}

		cases = append(cases, ModularityGainCase{
			Name:       rc.Name,
			Graph:      g,
			I:          in.I,
			Cluster:    in.Cluster,
			Partition:  in.Partition,
			Resolution: in.Resolution,
			DeltaQ:     out.DeltaQ,
		})
	}

	return cases, nil
}

// ------------ raw parser for "# / JSON / @ / JSON" ------------

type modularityGainRawCase struct {
	Name       string
	InputLine  string
	OutputLine string
}

func parseModularityGainRawCases(path string) ([]modularityGainRawCase, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var (
		cases            []modularityGainRawCase
		currentName      string
		pendingInput     string
		waitingForInput  bool
		waitingForOutput bool
	)

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		switch {
		case strings.HasPrefix(line, "#"):
			currentName = strings.TrimSpace(strings.TrimPrefix(line, "#"))
			pendingInput = ""
			waitingForInput = true
			waitingForOutput = false
		case strings.HasPrefix(line, "@"):
			waitingForOutput = true
		default:
			if waitingForInput {
				pendingInput = line
				waitingForInput = false
			} else if waitingForOutput {
				cases = append(cases, modularityGainRawCase{
					Name:       currentName,
					InputLine:  pendingInput,
					OutputLine: line,
				})
				waitingForOutput = false
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return cases, nil
}

func TestModularityGain_FromFile(t *testing.T) {
	const path = "clustering_tests/ModularityGain.txt" // adjust if the file is in a subdir

	cases, err := loadModularityGainCases(path)
	if err != nil {
		t.Fatalf("failed to load ModularityGain cases: %v", err)
	}

	const eps = 1e-9 // or reuse a global epsilon if you already have one

	for i, tc := range cases {
		tc := tc // capture range variable
		t.Run(tcName2(i, tc.Name), func(t *testing.T) {
			// Build nodesByCluster map from the partition
			nodesByCluster := tc.Graph.NodesByCluster(tc.Partition)
			got := tc.Graph.ModularityGain(tc.I, tc.Cluster, tc.Partition, tc.Resolution, nodesByCluster)

			if !floatEquals(got, tc.DeltaQ, eps) {
				t.Fatalf(
					"ModularityGain(i=%d, cluster=%d, partition=%v, resolution=%.4f) = %v, want %v",
					tc.I, tc.Cluster, tc.Partition, tc.Resolution,
					got, tc.DeltaQ,
				)
			}
		})
	}
}

// parser for Aggregate

// AggregateCase represents a single test case for (*Graph).Aggregate.
type AggregateCase struct {
	Name          string
	OriginalGraph *Graph
	Partition     []int
	ExpectedGraph *Graph
}

// loadAggregateCases reads a file with blocks:
//
//	# Case name
//	{"nodes":...,"edges":{...},"totalWeight":...,"partition":[...]}
//	@ some comment
//	{"nodes":...,"edges":{...},"totalWeight":...}
//
// and returns all parsed cases.
func loadAggregateCases(path string) ([]AggregateCase, error) {
	rawCases, err := parseAggregateRawCases(path)
	if err != nil {
		return nil, err
	}

	// JSON shapes for input / output.
	type graphInput struct {
		Nodes       int                `json:"nodes"`
		Edges       map[int][]EdgeJSON `json:"edges"`
		TotalWeight float64            `json:"totalWeight"`
		Partition   []int              `json:"partition"`
	}
	type graphOutput struct {
		Nodes       int                `json:"nodes"`
		Edges       map[int][]EdgeJSON `json:"edges"`
		TotalWeight float64            `json:"totalWeight"`
	}

	cases := make([]AggregateCase, 0, len(rawCases))
	for _, rc := range rawCases {
		var in graphInput
		if err := json.Unmarshal([]byte(rc.InputLine), &in); err != nil {
			return nil, err
		}

		var out graphOutput
		if err := json.Unmarshal([]byte(rc.OutputLine), &out); err != nil {
			return nil, err
		}

		// Build original graph
		origEdges := make(map[int][]Edge, len(in.Edges))
		for from, ejs := range in.Edges {
			es := make([]Edge, len(ejs))
			for i, ej := range ejs {
				es[i] = Edge{
					To:     ej.To,
					Weight: ej.Weight,
				}
			}
			origEdges[from] = es
		}
		originalGraph := &Graph{
			Nodes:       in.Nodes,
			Edges:       origEdges,
			TotalWeight: in.TotalWeight,
		}

		// Build expected aggregated graph
		expEdges := make(map[int][]Edge, len(out.Edges))
		for from, ejs := range out.Edges {
			es := make([]Edge, len(ejs))
			for i, ej := range ejs {
				es[i] = Edge{
					To:     ej.To,
					Weight: ej.Weight,
				}
			}
			expEdges[from] = es
		}
		expectedGraph := &Graph{
			Nodes:       out.Nodes,
			Edges:       expEdges,
			TotalWeight: out.TotalWeight,
		}

		cases = append(cases, AggregateCase{
			Name:          rc.Name,
			OriginalGraph: originalGraph,
			Partition:     in.Partition,
			ExpectedGraph: expectedGraph,
		})
	}

	return cases, nil
}

// -------- raw parser for "# / JSON / @ / JSON" --------

type aggregateRawCase struct {
	Name       string
	InputLine  string
	OutputLine string
}

func parseAggregateRawCases(path string) ([]aggregateRawCase, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var (
		cases            []aggregateRawCase
		currentName      string
		pendingInput     string
		waitingForInput  bool
		waitingForOutput bool
	)

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		switch {
		case strings.HasPrefix(line, "#"):
			// start a new case
			currentName = strings.TrimSpace(strings.TrimPrefix(line, "#"))
			pendingInput = ""
			waitingForInput = true
			waitingForOutput = false

		case strings.HasPrefix(line, "@"):
			// comment between input and output
			waitingForOutput = true

		default:
			if waitingForInput {
				// input JSON
				pendingInput = line
				waitingForInput = false
			} else if waitingForOutput {
				// output JSON → finalize case
				cases = append(cases, aggregateRawCase{
					Name:       currentName,
					InputLine:  pendingInput,
					OutputLine: line,
				})
				waitingForOutput = false
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return cases, nil
}

func TestAggregate_FromFile(t *testing.T) {
	const path = "clustering_tests/Aggregate.txt" // adjust if your file is in a subdir

	cases, err := loadAggregateCases(path)
	if err != nil {
		t.Fatalf("failed to load Aggregate cases: %v", err)
	}

	const eps = 1e-9 // or use your global epsilon if you already have one

	for i, tc := range cases {
		tc := tc // capture range variable
		t.Run(tcName2(i, tc.Name), func(t *testing.T) {
			got := tc.OriginalGraph.Aggregate(tc.Partition)

			if !graphsEqual(got, tc.ExpectedGraph, eps) {
				t.Fatalf("Aggregate(%v) = %#v, want %#v",
					tc.Partition, got, tc.ExpectedGraph)
			}
		})
	}
}

// graphsEqual compares two *Graph values (nodes, totalWeight, and edges).
func graphsEqual(a, b *Graph, eps float64) bool {
	if a == nil || b == nil {
		return a == b
	}
	if a.Nodes != b.Nodes {
		return false
	}
	if !floatEquals(a.TotalWeight, b.TotalWeight, eps) {
		return false
	}
	if len(a.Edges) != len(b.Edges) {
		return false
	}

	for from, expEdges := range b.Edges {
		gotEdges, ok := a.Edges[from]
		if !ok {
			return false
		}
		if !edgeSlicesEqual(gotEdges, expEdges, eps) {
			return false
		}
	}

	return true
}

// edgeSlicesEqual compares two []Edge slices with order and weight tolerance.
func edgeSlicesEqual(a, b []Edge, eps float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].To != b[i].To {
			return false
		}
		if !floatEquals(a[i].Weight, b[i].Weight, eps) {
			return false
		}
	}
	return true
}

// parser for Copy

// CopyCase is one fully parsed test case for Copy.
type CopyCase struct {
	Name            string
	InputPartition  []int
	OutputPartition []int
}

// loadCopyCases reads a file with blocks:
//
//	# Case name
//	{"partition":[...]}
//	@ some comment
//	{"partition":[...]}   // expected copied slice
//
// and returns all parsed cases.
func loadCopyCases(path string) ([]CopyCase, error) {
	rawCases, err := parseCopyRawCases(path)
	if err != nil {
		return nil, err
	}

	type input struct {
		Partition []int `json:"partition"`
	}
	type output struct {
		Partition []int `json:"partition"`
	}

	cases := make([]CopyCase, 0, len(rawCases))
	for _, rc := range rawCases {
		var in input
		if err := json.Unmarshal([]byte(rc.InputLine), &in); err != nil {
			return nil, err
		}

		var out output
		if err := json.Unmarshal([]byte(rc.OutputLine), &out); err != nil {
			return nil, err
		}

		cases = append(cases, CopyCase{
			Name:            rc.Name,
			InputPartition:  in.Partition,
			OutputPartition: out.Partition,
		})
	}

	return cases, nil
}

// ---------- internal raw parser for "# / JSON / @ / JSON" ----------

type copyRawCase struct {
	Name       string
	InputLine  string
	OutputLine string
}

func parseCopyRawCases(path string) ([]copyRawCase, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var (
		cases            []copyRawCase
		currentName      string
		pendingInput     string
		waitingForInput  bool
		waitingForOutput bool
	)

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		switch {
		case strings.HasPrefix(line, "#"):
			// start a new case
			currentName = strings.TrimSpace(strings.TrimPrefix(line, "#"))
			pendingInput = ""
			waitingForInput = true
			waitingForOutput = false

		case strings.HasPrefix(line, "@"):
			// comment between input and output
			waitingForOutput = true

		default:
			if waitingForInput {
				// this is the input JSON line
				pendingInput = line
				waitingForInput = false
			} else if waitingForOutput {
				// this is the output JSON line → finalize case
				cases = append(cases, copyRawCase{
					Name:       currentName,
					InputLine:  pendingInput,
					OutputLine: line,
				})
				waitingForOutput = false
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return cases, nil
}

func TestCopy_FromFile(t *testing.T) {
	const path = "clustering_tests/Copy.txt" // adjust if the file is in a subdir

	cases, err := loadCopyCases(path)
	if err != nil {
		t.Fatalf("failed to load Copy cases: %v", err)
	}

	for i, tc := range cases {
		tc := tc // capture range variable
		t.Run(tcName2(i, tc.Name), func(t *testing.T) {
			// keep a separate copy of the input to test aliasing later
			orig := append([]int(nil), tc.InputPartition...)

			got := Copy(tc.InputPartition)

			// 1) Check values are correct.
			if !equalIntSlices(got, tc.OutputPartition) {
				t.Fatalf("Copy(%v) = %v, want %v",
					tc.InputPartition, got, tc.OutputPartition)
			}

			// 2) Check deep copy: modifying got should NOT change tc.InputPartition.
			if len(got) > 0 {
				got[0]++
				if len(tc.InputPartition) > 0 && got[0] == tc.InputPartition[0] {
					t.Fatalf("Copy returned a slice sharing backing array with input; mutating result changed input")
				}
				// also ensure we didn't accidentally mutate our saved original
				if len(orig) > 0 && tc.InputPartition[0] != orig[0] {
					t.Fatalf("test mutated the original input slice unexpectedly")
				}
			}
		})
	}
}

//parser for Compare

// CompareCase is one fully parsed test case for Compare.
type CompareCase struct {
	Name  string
	A     []int
	B     []int
	Equal bool
}

// loadCompareCases reads a file with blocks:
//
//	# Case name
//	{"a":[...],"b":[...]}
//	@ some comment
//	{"equal":<bool>}
//
// and returns all parsed cases.
func loadCompareCases(path string) ([]CompareCase, error) {
	rawCases, err := parseCompareRawCases(path)
	if err != nil {
		return nil, err
	}

	type input struct {
		A []int `json:"a"`
		B []int `json:"b"`
	}
	type output struct {
		Equal bool `json:"equal"`
	}

	cases := make([]CompareCase, 0, len(rawCases))
	for _, rc := range rawCases {
		var in input
		if err := json.Unmarshal([]byte(rc.InputLine), &in); err != nil {
			return nil, err
		}

		var out output
		if err := json.Unmarshal([]byte(rc.OutputLine), &out); err != nil {
			return nil, err
		}

		cases = append(cases, CompareCase{
			Name:  rc.Name,
			A:     in.A,
			B:     in.B,
			Equal: out.Equal,
		})
	}

	return cases, nil
}

// ---------- internal raw parser for "# / JSON / @ / JSON" ----------

type compareRawCase struct {
	Name       string
	InputLine  string
	OutputLine string
}

func parseCompareRawCases(path string) ([]compareRawCase, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var (
		cases            []compareRawCase
		currentName      string
		pendingInput     string
		waitingForInput  bool
		waitingForOutput bool
	)

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		switch {
		case strings.HasPrefix(line, "#"):
			// start a new case
			currentName = strings.TrimSpace(strings.TrimPrefix(line, "#"))
			pendingInput = ""
			waitingForInput = true
			waitingForOutput = false

		case strings.HasPrefix(line, "@"):
			// comment between input and output
			waitingForOutput = true

		default:
			if waitingForInput {
				// this is the input JSON line
				pendingInput = line
				waitingForInput = false
			} else if waitingForOutput {
				// this is the output JSON line → finalize case
				cases = append(cases, compareRawCase{
					Name:       currentName,
					InputLine:  pendingInput,
					OutputLine: line,
				})
				waitingForOutput = false
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return cases, nil
}

func TestCompare_FromFile(t *testing.T) {
	const path = "clustering_tests/Compare.txt" // adjust if it's in a subdir

	cases, err := loadCompareCases(path)
	if err != nil {
		t.Fatalf("failed to load Compare cases: %v", err)
	}

	for i, tc := range cases {
		tc := tc // capture range variable
		t.Run(tcName2(i, tc.Name), func(t *testing.T) {
			got := Compare(tc.A, tc.B)
			if got != tc.Equal {
				t.Fatalf("Compare(%v, %v) = %v, want %v",
					tc.A, tc.B, got, tc.Equal)
			}
		})
	}
}
