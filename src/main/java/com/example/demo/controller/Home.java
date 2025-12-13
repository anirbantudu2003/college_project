package com.example.demo.controller;

import com.example.demo.service.GeminiService;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/gemini")
public class Home {

    private final GeminiService geminiService;

    public Home(GeminiService geminiService) {
        this.geminiService = geminiService;
    }

    @PostMapping("/generate")
    public String generate(@RequestBody Prompt prompt) {
        return geminiService.generateText(prompt.getText());
    }

    public static class Prompt {
        private String text;
		
        public String getText() { return text; }
        public void setText(String text) { this.text = text; }
    }
}
