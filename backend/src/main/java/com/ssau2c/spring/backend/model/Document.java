package com.ssau2c.spring.backend.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.*;

import java.util.List;


@Data
@Builder
public class Document {
    private int id;
    private List<Text> text;

    public Document(@JsonProperty("id") int id, @JsonProperty("texts") List<Text> text) {
        this.id = id;
        this.text = text;
    }
}
