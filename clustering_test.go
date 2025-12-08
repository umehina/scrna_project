//Name: Yinan Zhu 
//Date added: 12/06/2025 and 12/07/2025
// Disclaimer: we consulted ChatGPT for many parser functions in this file.

// distance_matrix_test.go
package main

import (
	"bufio"
	"encoding/json"
	"math"
	"os"
	"strings"
	"testing"

	"gonum.org/v1/gonum/mat"
)

// ---------- generic parser for # / @ format ----------

// ioCase holds one test case parsed from the .txt file:
//
//	# Name
//	{"input": ...}
//	@ comment
//	{"output": ...}
type ioCase struct {
	Name       string
	InputLine  string
	OutputLine string
}

// loadIOCasesFromFile parses a text file in the "# / @ / JSON" format
// and returns all cases.
func loadIOCasesFromFile(path string) ([]ioCase, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var (
		cases            []ioCase
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
				cases = append(cases, ioCase{
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
			got, err := DistanceMatrix(data)

			// Handle error-expected cases
			if out.WantErr != "" {
				if err == nil {
					t.Fatalf("expected error containing %q, got nil", out.WantErr)
				}
				if !strings.Contains(err.Error(), out.WantErr) {
					t.Fatalf("error %q does not contain expected substring %q", err.Error(), out.WantErr)
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error from DistanceMatrix: %v", err)
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


// ---------- generic # / @ / JSON parser ----------


// loadIOCasesFromFile parses a text file with repeated blocks:
//
//   # Case name
//   <input JSON>
//   @ comment (ignored for parsing)
//   <output JSON>

// ---------- specialized parser for fillDirectedKNNWeights ----------

// knnWeightsInput matches the *input* JSON:
//
//   {"k":1,"dist":[[0,1,2],[1,0,3],[2,3,0]]}
//
type knnWeightsInput struct {
	K    int         `json:"k"`
	Dist [][]float64 `json:"dist"`
}

// knnWeightsOutput matches the *output* JSON:
//
//   {"weights":[[0,0.5,0],[0.5,0,0],[0.3333333,0,0]]}
//
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
//   {"first":[...], "second":[...]}
//
type euclideanInput struct {
	First  []float64 `json:"first"`
	Second []float64 `json:"second"`
}

// euclideanOutput matches the *output* JSON:
//
//   {"dist":5}
//
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

// ---------- parser for buildKNNGraph ----------
// loadIOCasesFromFile parses a text file with repeated blocks:
//
//   # Case name
//   <input JSON>
//   @ comment (ignored for parsing)
//   <output JSON>
//

// ---------- specialized structs for BuildKNNGraph ----------

// knnGraphInput matches the *input* JSON:
//
//   {"k":1,"dist":[[0,1,2],[1,0,3],[2,3,0]]}
//
type knnGraphInput struct {
	K    int         `json:"k"`
	Dist [][]float64 `json:"dist"`
}

// expectedEdge is how we encode one edge in JSON:
//
//   {"to":1,"weight":1.0}
//
type expectedEdge struct {
	To     int     `json:"to"`
	Weight float64 `json:"weight"`
}

// knnGraphOutput matches the *expected* Graph:
//
//   {"nodes":3,"edges":{"0":[{"to":1,...}]}, "totalWeight":2.1666667}
//
type knnGraphOutput struct {
	Nodes       int                      `json:"nodes"`
	Edges       map[int][]expectedEdge   `json:"edges"`
	TotalWeight float64                  `json:"totalWeight"`
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

// ---------- parser for DirectedKNNWeight ----------



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
	Name  string
	Row   []float64
	R     int
	Idx   []int
	Dist  []float64
}

// Internal raw representation of each block in the txt file.
type getNeighborsRowRawCase struct {
	Name       string
	InputLine  string
	OutputLine string
}

// loadGetNeighborsRowCases reads a file with blocks in this pattern:
//
//   # Case name
//   {"row":[...],"r":1}
//   @ some comment
//   {"idx":[...],"dist":[...]}
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
//   # Case name
//   {"k":2,"idx":[0,1,2],"dist":[5,1,3]}
//   @ some comment
//   {"idx":[1,2],"dist":[1,3]}
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

// ---------- parser for distanceToWeights ----------


// DistanceToWeightCase is one fully-parsed test case.
type DistanceToWeightCase struct {
	Name    string
	Distance float64
	Weight   float64
}

// loadDistanceToWeightCases reads a file with blocks like:
//
//   # Case name
//   {"distance": ...}
//   @ some comment
//   {"weight": ...}
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
//   # Case name
//   {"directed":[[...], ...]}
//   @ some comment
//   {"edges":{...},"totalWeight":...}
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
//   # Case name
//   {"nodes":...,"edges":{...},"totalWeight":...,"resolution":...,"gamma":...,"theta":...,"maxIter":...}
//   @ some comment
//   {"clusters":[...]}
//
// and returns all parsed cases.
func loadLeidenCases(path string) ([]LeidenCase, error) {
	rawCases, err := parseLeidenRawCases(path)
	if err != nil {
		return nil, err
	}

	// JSON shapes for input / output
	type input struct {
		Nodes       int                 `json:"nodes"`
		Edges       map[int][]EdgeJSON  `json:"edges"`
		TotalWeight float64             `json:"totalWeight"`
		Resolution  float64             `json:"resolution"`
		Gamma       float64             `json:"gamma"`
		Theta       float64             `json:"theta"`
		MaxIter     int                 `json:"maxIter"`
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

// ---------- parser for Refine ----------

// RefineCase is one fully parsed test case for Graph.Refine.
type RefineCase struct {
	Name      string
	Partition []int
	Clusters  []int
}

// loadRefineCases reads a file with blocks:
//
//   # Case name
//   {"partition":[...]}
//   @ some comment
//   {"clusters":[...]}
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
//   # Case name
//   {"nodes":...,"edges":{...},"totalWeight":...,"partition":[...],
//    "resolution":...,"gamma":...,"theta":...}
//   @ some comment
//   {"clusters":[...]}
//
// and returns all parsed cases.
func loadRefinePartitionCases(path string) ([]RefinePartitionCase, error) {
	rawCases, err := parseRefinePartitionRawCases(path)
	if err != nil {
		return nil, err
	}

	// JSON shapes for input/output
	type input struct {
		Nodes       int                 `json:"nodes"`
		Edges       map[int][]EdgeJSON  `json:"edges"`
		TotalWeight float64             `json:"totalWeight"`
		Partition   []int               `json:"partition"`
		Resolution  float64             `json:"resolution"`
		Gamma       float64             `json:"gamma"`
		Theta       float64             `json:"theta"`
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
		Nodes       int                 `json:"nodes"`
		Edges       map[int][]EdgeJSON  `json:"edges"`
		TotalWeight float64             `json:"totalWeight"`
		Partition   []int               `json:"partition"`
		Resolution  float64             `json:"resolution"`
		Gamma       float64             `json:"gamma"`
		Theta       float64             `json:"theta"`
	}
	type output struct {
		Clusters []int `json:"clusters"`
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
//   # Case name
//   {"n":...}
//   @ some comment
//   {"partition":[...]}
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
//   # Case name
//   {"nodes":...,"edges":{...},"totalWeight":...}
//   @ some comment
//   {"partition":[...]}
//
// and returns all parsed cases.
func loadGraphInitSingletonPartitionCases(path string) ([]GraphInitSingletonPartitionCase, error) {
	rawCases, err := parseGraphInitSingletonPartitionRawCases(path)
	if err != nil {
		return nil, err
	}

	// JSON shapes for input/output.
	type input struct {
		Nodes       int                 `json:"nodes"`
		Edges       map[int][]EdgeJSON  `json:"edges"`
		TotalWeight float64             `json:"totalWeight"`
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

// parser for NodesByCluster

// NodesByClusterCase is one fully parsed test case for NodesByCluster.
type NodesByClusterCase struct {
	Name      string
	Partition []int
	Groups    map[int][]int
}

// loadNodesByClusterCases reads a file with blocks:
//
//   # Case name
//   {"partition":[...]}
//   @ some comment
//   {"groups":{"0":[...], "1":[...]}}
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

//parser for refinedPartition


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
//   # Case name
//   {"nodes":[...],"partition":[...],"refinedPartition":[...],
//    "resolution":...,"gamma":...,"theta":...}
//   @ some comment
//   {"partition":[...]}    // expected refined partition
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

// parser for FindWellConnectedClusters

// EdgeJSON is a lightweight version of Edge for JSON decoding.


// FindWellConnectedClustersCase is one fully parsed test case
// for Graph.FindWellConnectedClusters.
type FindWellConnectedClustersCase struct {
	Name            string
	Graph           *Graph
	Subset          []int
	RefinedPartition []int
	Gamma           float64
	Clusters        []int
}

// loadFindWellConnectedClustersCases reads a file with blocks:
//
//   # Case name
//   {"nodes":...,"edges":{...},"totalWeight":...,
//    "subset":[...],"refinedPartition":[...],"gamma":...}
//   @ some comment
//   {"clusters":[...]}
//
// and returns all parsed cases.
func loadFindWellConnectedClustersCases(path string) ([]FindWellConnectedClustersCase, error) {
	rawCases, err := parseFindWellConnectedClustersRawCases(path)
	if err != nil {
		return nil, err
	}

	// JSON shapes for input/output.
	type input struct {
		Nodes            int                  `json:"nodes"`
		Edges            map[int][]EdgeJSON   `json:"edges"`
		TotalWeight      float64              `json:"totalWeight"`
		Subset           []int                `json:"subset"`
		RefinedPartition []int                `json:"refinedPartition"`
		Gamma            float64              `json:"gamma"`
	}
	type output struct {
		Clusters []int `json:"clusters"`
	}

	cases := make([]FindWellConnectedClustersCase, 0, len(rawCases))
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

		cases = append(cases, FindWellConnectedClustersCase{
			Name:             rc.Name,
			Graph:            g,
			Subset:           in.Subset,
			RefinedPartition: in.RefinedPartition,
			Gamma:            in.Gamma,
			Clusters:         out.Clusters,
		})
	}

	return cases, nil
}

// ---------- internal raw parser for "# / JSON / @ / JSON" ----------

type findWellConnectedClustersRawCase struct {
	Name       string
	InputLine  string
	OutputLine string
}

func parseFindWellConnectedClustersRawCases(path string) ([]findWellConnectedClustersRawCase, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var (
		cases            []findWellConnectedClustersRawCase
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
				cases = append(cases, findWellConnectedClustersRawCase{
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

//parser for SampleCommunity

// SampleCommunityCase is one fully parsed test case
// for Graph.SampleCommunity (receiver g is unused in the function body).
type SampleCommunityCase struct {
	Name      string
	Clusters  []int
	Probs     map[int]float64
	Selected  int
}

// loadSampleCommunityCases reads a file with blocks:
//
//   # Case name
//   {"clusters":[...],"probs":{"0":1.0,...}}
//   @ some comment
//   {"selected":<int>}
//
// and returns all parsed cases.
func loadSampleCommunityCases(path string) ([]SampleCommunityCase, error) {
	rawCases, err := parseSampleCommunityRawCases(path)
	if err != nil {
		return nil, err
	}

	// JSON shapes for input/output.
	type input struct {
		Clusters []int              `json:"clusters"`
		Probs    map[int]float64    `json:"probs"`
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

// parser for ComputeMoveProbabilities 

// ComputeMoveProbabilityCase represents one fully parsed test case
// for Graph.ComputeMoveProbability.
type ComputeMoveProbabilityCase struct {
	Name             string
	CurrNode         int
	CandidateClusters []int
	RefinedPartition []int
	Theta            float64
	Resolution       float64
	Probs            map[int]float64
}

// loadComputeMoveProbabilityCases reads a file with blocks:
//
//   # Case name
//   {"currNode":...,"candidateClusters":[...],"refinedPartition":[...],
//    "theta":...,"resolution":...}
//   @ some comment
//   {"probs":{"clusterID":prob, ...}}
//
// and returns all parsed cases.
func loadComputeMoveProbabilityCases(path string) ([]ComputeMoveProbabilityCase, error) {
	rawCases, err := parseComputeMoveProbabilityRawCases(path)
	if err != nil {
		return nil, err
	}

	// JSON shapes for input/output.
	type input struct {
		CurrNode          int       `json:"currNode"`
		CandidateClusters []int     `json:"candidateClusters"`
		RefinedPartition  []int     `json:"refinedPartition"`
		Theta             float64   `json:"theta"`
		Resolution        float64   `json:"resolution"`
	}
	type output struct {
		Probs map[int]float64 `json:"probs"`
	}

	cases := make([]ComputeMoveProbabilityCase, 0, len(rawCases))
	for _, rc := range rawCases {
		var in input
		if err := json.Unmarshal([]byte(rc.InputLine), &in); err != nil {
			return nil, err
		}

		var out output
		if err := json.Unmarshal([]byte(rc.OutputLine), &out); err != nil {
			return nil, err
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

// ---------- internal raw parser for "# / JSON / @ / JSON" ----------

type computeMoveProbabilityRawCase struct {
	Name       string
	InputLine  string
	OutputLine string
}

func parseComputeMoveProbabilityRawCases(path string) ([]computeMoveProbabilityRawCase, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var (
		cases            []computeMoveProbabilityRawCase
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
				cases = append(cases, computeMoveProbabilityRawCase{
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

// parser for FindWellConnectedClusters


// FindWellConnectedNodesCase is one fully parsed test case
// for Graph.FindWellConnectedNodes.
type FindWellConnectedNodesCase struct {
	Name       string
	Graph      *Graph
	Subset     []int
	Partition  []int
	Gamma      float64
	Connected  []int
}

// loadFindWellConnectedNodesCases reads a file with blocks:
//
//   # Case name
//   {"graphNodes":...,"edges":{...},"totalWeight":...,
//    "subset":[...],"partition":[...],"gamma":...}
//   @ some comment
//   {"connected":[...]}
//
// and returns all parsed cases.
func loadFindWellConnectedNodesCases(path string) ([]FindWellConnectedNodesCase, error) {
	rawCases, err := parseFindWellConnectedNodesRawCases(path)
	if err != nil {
		return nil, err
	}

	// JSON shapes for input / output.
	type input struct {
		GraphNodes int                  `json:"graphNodes"`
		Edges      map[int][]EdgeJSON   `json:"edges"`
		TotalWeight float64             `json:"totalWeight"`
		Subset     []int                `json:"subset"`
		Partition  []int                `json:"partition"`
		Gamma      float64              `json:"gamma"`
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
//   # Case name
//   {"nodes":...,"edges":{...},"totalWeight":...,"node":<int>,"cluster":[...]}
//   @ some comment
//   {"sum":<float>}
//
// and returns all parsed cases.
func loadEdgesToClusterCases(path string) ([]EdgesToClusterCase, error) {
	rawCases, err := parseEdgesToClusterRawCases(path)
	if err != nil {
		return nil, err
	}

	// JSON shapes for input/output.
	type input struct {
		Nodes       int                 `json:"nodes"`
		Edges       map[int][]EdgeJSON  `json:"edges"`
		TotalWeight float64             `json:"totalWeight"`
		Node        int                 `json:"node"`
		Cluster     []int               `json:"cluster"`
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
//   # Case name
//   {"nodes":...,"edges":{...},"totalWeight":...,"cluster":[...]}
//   @ some comment
//   {"degree":<float>}
//
// and returns all parsed cases.
func loadClusterDegreeCases(path string) ([]ClusterDegreeCase, error) {
	rawCases, err := parseClusterDegreeRawCases(path)
	if err != nil {
		return nil, err
	}

	// JSON shapes for input / output.
	type input struct {
		Nodes       int                 `json:"nodes"`
		Edges       map[int][]EdgeJSON  `json:"edges"`
		TotalWeight float64             `json:"totalWeight"`
		Cluster     []int               `json:"cluster"`
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
//   # Case name
//   {"node":<int>,"partition":[...]}
//   @ some comment
//   {"singleton":<bool>}
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
//   # Case name
//   {"nodes":...,"edges":{...},"totalWeight":...,"partition":[...],"resolution":...}
//   @ some comment
//   {"partition":[...]}   // expected final partition
//
// and returns all parsed cases.
func loadMoveNodesCases(path string) ([]MoveNodesCase, error) {
	rawCases, err := parseMoveNodesRawCases(path)
	if err != nil {
		return nil, err
	}

	// JSON shapes for input / output.
	type input struct {
		Nodes       int                 `json:"nodes"`
		Edges       map[int][]EdgeJSON  `json:"edges"`
		TotalWeight float64             `json:"totalWeight"`
		Partition   []int               `json:"partition"`
		Resolution  float64             `json:"resolution"`
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

//parser for FindBestClustering


// FindBestClusterCase represents one test case for FindBestCluster.
type FindBestClusterCase struct {
	Name             string
	Graph            *Graph
	Node             int
	CandidateClusters []int
	Partition        []int
	Resolution       float64
	BestCluster      int
	Improved         bool
}

// loadFindBestClusterCases reads a file with blocks:
//
//   # Case name
//   {"nodes":...,"edges":{...},"totalWeight":...,
//    "node":<int>,"candidateClusters":[...],
//    "partition":[...],"resolution":...}
//   @ some comment
//   {"bestCluster":<int>,"improved":<bool>}
//
// and returns all parsed cases.
func loadFindBestClusterCases(path string) ([]FindBestClusterCase, error) {
	rawCases, err := parseFindBestClusterRawCases(path)
	if err != nil {
		return nil, err
	}

	// JSON input/output shapes.
	type input struct {
		Nodes            int                 `json:"nodes"`
		Edges            map[int][]EdgeJSON  `json:"edges"`
		TotalWeight      float64             `json:"totalWeight"`
		Node             int                 `json:"node"`
		CandidateClusters []int              `json:"candidateClusters"`
		Partition        []int               `json:"partition"`
		Resolution       float64             `json:"resolution"`
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
			Name:             rc.Name,
			Graph:            g,
			Node:             in.Node,
			CandidateClusters: in.CandidateClusters,
			Partition:        in.Partition,
			Resolution:       in.Resolution,
			BestCluster:      out.BestCluster,
			Improved:         out.Improved,
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
//   # Case name
//   {"n":<int>,"seed":<int>}
//   @ some comment
//   {"expectedLen":<int>}
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
//   # Case name
//   {"node":...,"edges":[{"to":...,"weight":...},...],"partition":[...]}
//   @ some comment
//   {"candidates":[...]}
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

//parser for ModularityGain

// ModularityGainCase is one test case for Graph.ModularityGain.
type ModularityGainCase struct {
	Name        string
	Graph       *Graph
	I           int
	Cluster     int
	Partition   []int
	Resolution  float64
	DeltaQ      float64
}

// loadModularityGainCases reads:
//
//   # Case name
//   {"nodes":...,"edges":{...},"totalWeight":...,"i":...,"cluster":...,
//    "partition":[...],"resolution":...}
//   @ comment
//   {"deltaQ":...}
func loadModularityGainCases(path string) ([]ModularityGainCase, error) {
	rawCases, err := parseModularityGainRawCases(path)
	if err != nil {
		return nil, err
	}

	type input struct {
		Nodes       int                 `json:"nodes"`
		Edges       map[int][]EdgeJSON  `json:"edges"`
		TotalWeight float64             `json:"totalWeight"`
		I           int                 `json:"i"`
		Cluster     int                 `json:"cluster"`
		Partition   []int               `json:"partition"`
		Resolution  float64             `json:"resolution"`
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
//   # Case name
//   {"nodes":...,"edges":{...},"totalWeight":...,"partition":[...]}
//   @ some comment
//   {"nodes":...,"edges":{...},"totalWeight":...}
//
// and returns all parsed cases.
func loadAggregateCases(path string) ([]AggregateCase, error) {
	rawCases, err := parseAggregateRawCases(path)
	if err != nil {
		return nil, err
	}

	// JSON shapes for input / output.
	type graphInput struct {
		Nodes       int                 `json:"nodes"`
		Edges       map[int][]EdgeJSON  `json:"edges"`
		TotalWeight float64             `json:"totalWeight"`
		Partition   []int               `json:"partition"`
	}
	type graphOutput struct {
		Nodes       int                 `json:"nodes"`
		Edges       map[int][]EdgeJSON  `json:"edges"`
		TotalWeight float64             `json:"totalWeight"`
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

// parser for Copy


// CopyCase is one fully parsed test case for Copy.
type CopyCase struct {
	Name            string
	InputPartition  []int
	OutputPartition []int
}

// loadCopyCases reads a file with blocks:
//
//   # Case name
//   {"partition":[...]}
//   @ some comment
//   {"partition":[...]}   // expected copied slice
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

//parser for Compare

// CompareCase is one fully parsed test case for Compare.
type CompareCase struct {
	Name   string
	A      []int
	B      []int
	Equal  bool
}

// loadCompareCases reads a file with blocks:
//
//   # Case name
//   {"a":[...],"b":[...]}
//   @ some comment
//   {"equal":<bool>}
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