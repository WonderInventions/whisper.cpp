package whisper_test

import (
	"io"
	"os"
	"sync"
	"testing"

	// Packages
	whisper "github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper"
	"github.com/go-audio/wav"
	assert "github.com/stretchr/testify/assert"
)

const (
	ModelPath  = "../../models/ggml-tiny.bin"
	SamplePath = "../../samples/jfk.wav"
)

func Test_Whisper_000(t *testing.T) {
	assert := assert.New(t)
	if _, err := os.Stat(ModelPath); os.IsNotExist(err) {
		t.Skip("Skipping test, model not found:", ModelPath)
	}
	if _, err := os.Stat(SamplePath); os.IsNotExist(err) {
		t.Skip("Skipping test, sample not found:", SamplePath)
	}

	// Load model
	model, err := whisper.New(ModelPath)
	assert.NoError(err)
	assert.NotNil(model)
	assert.NoError(model.Close())

	t.Log("languages=", model.Languages())
}

func Test_Whisper_001(t *testing.T) {
	assert := assert.New(t)
	if _, err := os.Stat(ModelPath); os.IsNotExist(err) {
		t.Skip("Skipping test, model not found:", ModelPath)
	}
	if _, err := os.Stat(SamplePath); os.IsNotExist(err) {
		t.Skip("Skipping test, sample not found:", SamplePath)
	}

	// Load model
	model, err := whisper.New(ModelPath)
	assert.NoError(err)
	assert.NotNil(model)
	defer model.Close()

	// Get context for decoding
	ctx, err := model.NewContext()
	assert.NoError(err)
	assert.NotNil(ctx)

}

func Test_Whisper_003(t *testing.T) {
	assert := assert.New(t)
	if _, err := os.Stat(ModelPath); os.IsNotExist(err) {
		t.Skip("Skipping test, model not found:", ModelPath)
	}
	if _, err := os.Stat(SamplePath); os.IsNotExist(err) {
		t.Skip("Skipping test, sample not found:", SamplePath)
	}

	// Load model
	model, err := whisper.New(ModelPath)
	assert.NoError(err)
	assert.NotNil(model)
	defer model.Close()

	// Get context for decoding
	ctx, err := model.NewContext()
	assert.NoError(err)
	assert.NotNil(ctx)

	// Get the wav file.
	rdr, err := os.Open(SamplePath)
	assert.NoError(err)
	dec := wav.NewDecoder(rdr)
	buf, err := dec.FullPCMBuffer()
	assert.NoError(err)

	// Check that we can use a context more than once.
	{
		var firstResult string
		err = ctx.Process(buf.AsFloat32Buffer().Data, func(seg whisper.Segment) {
			t.Logf("text1: %v", seg.Text)
			firstResult = seg.Text
		})
		assert.NoError(err)
		assert.True(firstResult != "")
		err = ctx.Process(buf.AsFloat32Buffer().Data, func(seg whisper.Segment) {
			t.Logf("text2: %v", seg.Text)
			assert.Equal(firstResult, seg.Text)
		})
		assert.NoError(err)
	}

	// Now check that we can use the non-callback process.
	{
		var firstResult string
		err = ctx.Process(buf.AsFloat32Buffer().Data, nil)
		assert.NoError(err)
		for {
			seg, err := ctx.NextSegment()
			if err == io.EOF {
				break
			}
			t.Logf("text1: %v", seg.Text)
			firstResult += seg.Text
		}
		assert.True(firstResult != "")
		err = ctx.Process(buf.AsFloat32Buffer().Data, nil)
		assert.NoError(err)
		var secondResult string
		for {
			seg, err := ctx.NextSegment()
			if err == io.EOF {
				break
			}
			t.Logf("text2: %v", seg.Text)
			secondResult += seg.Text
		}
		assert.Equal(firstResult, secondResult)
	}
}

func processSample(t *testing.T, ctx whisper.Context, data []float32, wg *sync.WaitGroup) {
	err := ctx.Process(data, func(seg whisper.Segment) {
		t.Logf("text: %v", seg.Text)
	})
	assert.NoError(t, err)
	wg.Done()
}
