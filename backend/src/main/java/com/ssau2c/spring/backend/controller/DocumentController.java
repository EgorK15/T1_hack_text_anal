package com.ssau2c.spring.backend.controller;

import com.ssau2c.spring.backend.model.Document;
import com.ssau2c.spring.backend.repository.DataBaseMemoryDocumentDAO;
import com.ssau2c.spring.backend.service.DocumentService;
import com.ssau2c.spring.backend.service.impl.DataBaseDocumentServiceImpl;
import lombok.AllArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/v1/documents")
@AllArgsConstructor
public class DocumentController {

    private final DocumentService documentService = new DataBaseDocumentServiceImpl(new DataBaseMemoryDocumentDAO());

    @GetMapping
    public List<Document> findAllDocument() {
        return documentService.findAllDocument();
    }

    @PostMapping("/save")
    public String saveDocument(@RequestBody String texts) {
        documentService.saveDocument(texts);
        return "Document saved";
    }

    @GetMapping("/{id}")
    public Document findDocumentById(@PathVariable int id) {
        return documentService.findDocumentById(id);
    }

    @PutMapping("/update")
    public Document updateDocument(@RequestBody Document document) {
        return documentService.updateDocument(document);
    }

    @DeleteMapping("/delete/{id}")
    public void deleteDocumentById(@PathVariable int id) {
        documentService.deleteDocument(id);
    }
}
