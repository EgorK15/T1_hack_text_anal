package com.ssau2c.spring.backend.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class Text {

    private String content;
    private String coordinates;
    private String signature;

    public Text(@JsonProperty("content") String content, @JsonProperty("coordinates") String coordinates, @JsonProperty("signature") String signature) {
        this.content = content;
        this.coordinates = coordinates;
        this.signature = signature;
    }

    @Override
    public String toString() {
        return "{\"content\":\"" + content + "\", \"coordinates\":\"" + coordinates + "\", \"signature\":\"" + signature + "\"}";
    }
}
