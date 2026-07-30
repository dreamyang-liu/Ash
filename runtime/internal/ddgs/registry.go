package ddgs

// Engine registry, mirroring ddgs/engines/__init__.py's ENGINES dict.
// Used by ddgs.go (DDGS search methods) to resolve category+backend to
// engine factories. The "bing" text engine is disabled upstream
// (disabled=True) and is therefore not registered here either.
// User instruction: "帮我写一个golang的library，主要就是把
// https://github.com/deedy5/ddgs 的功能复刻出来，然后提供和python版本一致的接口".

import "time"

// engineFactory constructs an engine instance.
type engineFactory func(proxy string, timeout time.Duration, verify Verify) (searchEngine, error)

// engineRegistry maps category -> backend name -> factory.
// Mirrors ENGINES[category][name] = class in the Python implementation.
var engineRegistry = map[string]map[string]engineFactory{
	"text": {
		"brave":      newBrave,
		"duckduckgo": newDuckduckgoText,
		"google":     newGoogle,
		"grokipedia": newGrokipedia,
		"mojeek":     newMojeek,
		"startpage":  newStartpage,
		"wikipedia":  newWikipedia,
		"yahoo":      newYahoo,
		"yandex":     newYandex,
	},
	"images": {
		"bing":       newBingImages,
		"duckduckgo": newDuckduckgoImages,
	},
	"news": {
		"bing":       newBingNews,
		"duckduckgo": newDuckduckgoNews,
		"yahoo":      newYahooNews,
	},
	"videos": {
		"duckduckgo": newDuckduckgoVideos,
	},
	"books": {
		"annasarchive": newAnnasArchive,
	},
}
