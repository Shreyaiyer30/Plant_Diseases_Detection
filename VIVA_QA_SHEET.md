# PlantCure - Viva Q&A Sheet

Use these for viva preparation. Keep answers short and confident.

## Project Basics

**Q1. What is your project?**  
A: PlantCure is an AI-based web app that detects plant leaf diseases from image upload/camera capture and gives treatment guidance through an assistant chatbot.

**Q2. Why did you choose this topic?**  
A: Farmers often cannot identify diseases early. This system gives quick support for early diagnosis and action.

**Q3. What is the novelty in your project?**  
A: Integrated pipeline: authentication, real-time camera capture, disease prediction, invalid-image rejection, history analytics, and advisory chatbot in one app.

## Dataset and Model

**Q4. Which dataset did you use?**  
A: PlantVillage dataset (Kaggle), with class folders per crop-disease.

**Q5. Which model architecture did you use? Why?**  
A: MobileNetV2 transfer learning because it provides good accuracy with lower compute cost.

**Q6. How did you train your model?**  
A: Data augmentation + train/validation split + class weighting + fine-tuning + early stopping and LR reduction.

**Q7. Why class labels are important?**  
A: Inference class index must map to correct class name; `class_labels.json` preserves that mapping from training.

**Q8. What is label smoothing and why used?**  
A: It reduces overconfidence and helps generalization.

## System Design

**Q9. Explain end-to-end flow briefly.**  
A: User uploads/captures image -> backend preprocesses -> model predicts -> disease details returned -> result stored in DB -> history/dashboard updated.

**Q10. Which database and why?**  
A: SQLite for simplicity and portability in final-year project setup.

**Q11. How did you secure user data access?**  
A: Login-required routes; uploaded image serving checks file ownership via user-id prefix.

## Reliability and Edge Cases

**Q12. How do you handle wrong images (hand/background)?**  
A: Added non-leaf/low-confidence rejection using image heuristics and prediction uncertainty checks.

**Q13. What happens if chatbot API is unavailable?**  
A: App uses fallback rule-based responses so feature remains usable.

**Q14. How did you handle camera issues?**  
A: Added device enumeration, camera switching, black-frame detection, and capture validation.

## Evaluation

**Q15. Which metrics did you use?**  
A: Accuracy, loss, per-class precision/recall/F1, confusion matrix, and real-world app test cases.

**Q16. Why can model still misclassify?**  
A: Similar disease patterns, low-light/blur, unseen real-world backgrounds, and limited class coverage if dataset is partial.

## Limitations and Future Scope

**Q17. Current limitations?**  
A: Depends on image quality, dataset class coverage, and CPU training time.

**Q18. Future improvements?**  
A: Leaf segmentation, more real-field datasets, multilingual voice assistant, fertilizer recommendation DB, and cloud deployment.

## Demo Questions

**Q19. How do you prove model is not random?**  
A: Show training logs, saved model, class labels, and consistent results across multiple test samples.

**Q20. What if prediction confidence is low?**  
A: App asks user to re-upload a clearer leaf image instead of forcing a disease output.

---

## 60-Second Project Pitch

PlantCure is a practical AI system for early plant disease support.  
It takes leaf images via upload/camera, predicts disease using a trained MobileNetV2 model, rejects invalid images, and gives treatment/fertilizer guidance through a chatbot.  
It also stores diagnosis history and analytics per user.  
The project focuses on real usability for farmers, not just model training.
