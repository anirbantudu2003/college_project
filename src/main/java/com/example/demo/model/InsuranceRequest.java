package com.example.demo.model;

public class InsuranceRequest {
    private String name;
    private Integer age;
    private String gender;
    private Boolean smoker;
    private String location;
    private Integer kids;
    private Double bmi;
    private String existingCondition;
    private String model; // Model selection: random_forest, decision_tree, linear_regression

    public InsuranceRequest() {}

    public String getName() { return name; }
    public void setName(String name) { this.name = name; }

    public Integer getAge() { return age; }
    public void setAge(Integer age) { this.age = age; }

    public String getGender() { return gender; }
    public void setGender(String gender) { this.gender = gender; }

    public Boolean getSmoker() { return smoker; }
    public void setSmoker(Boolean smoker) { this.smoker = smoker; }

    public String getLocation() { return location; }
    public void setLocation(String location) { this.location = location; }

    public Integer getKids() { return kids; }
    public void setKids(Integer kids) { this.kids = kids; }

    public Double getBmi() { return bmi; }
    public void setBmi(Double bmi) { this.bmi = bmi; }

    public String getExistingCondition() { return existingCondition; }
    public void setExistingCondition(String existingCondition) { this.existingCondition = existingCondition; }

    public String getModel() { return model; }
    public void setModel(String model) { this.model = model; }
}
