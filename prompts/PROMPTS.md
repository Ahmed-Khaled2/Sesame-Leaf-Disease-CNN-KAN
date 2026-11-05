# Prompts used to generate the synthetic training images

This file holds the same content as Supplementary Material S1 of the paper *Sesame Leaf Disease Detection Using a Hybrid CNN-KAN Architecture*. The images themselves are in [`synthetic_images/`](../synthetic_images).

## 1. Tool, date and model

The synthetic Phyllody and Powdery Mildew images were generated on 11 April 2026 in the Gemini app (Google, https://gemini.google.com), in several separate chats. At that time the default image model of the Gemini app, for its Fast, Thinking and Pro modes, was Nano Banana 2 (Gemini 3.1 Flash Image) [1]. The app does not show the model version in the chat history, so the version could not be recorded more precisely. Some turns of the Phyllody chat used the Thinking mode.

The chat histories were exported after the experiments, and not every prompt was preserved. The prompts below are the ones that could be recovered. Spelling mistakes in the typed prompts were corrected; the JSON templates are given as Gemini returned them.

## 2. Procedure

1. A short starting prompt produced the first images (Sections 3.1 and 4.1).
2. Gemini was asked to describe a good output, or an uploaded photo, as a JSON prompt template with fields that can be changed. These templates were then pasted back with new values (Sections 3.2 to 3.4 and 4.2 to 4.4).
3. Follow-up prompts asked for four more images at a time, with changes in view (macro close-up, medium close-up, wider view), camera angle, lighting, background, leaf shape and infection level, and without any text or labels. Most outputs were therefore 2 x 2 grids of four images.
4. In one Phyllody chat, three real ERFD Phyllody photos were uploaded as examples ("These are some samples, can you generate more similar images?"). In the main study, no synthetic tile reached the near-copy threshold with any real image (highest cosine similarity of ResNet50 features 0.94; near-copy threshold 0.97).
5. Each grid was split into four tiles. The visible Gemini logo in the corner of the bottom-right tile was covered with a patch taken from just above it, so that the logo could not act as a class cue.
6. More than 500 images were generated in total. The first author removed tiles that did not look like sesame or did not match their class, cropped away unwanted details, and removed identical or highly similar tiles with a similarity check. 75 Phyllody tiles (from 18 generated images) and 111 Powdery Mildew tiles (from 62) remained.
7. The synthetic tiles were used in training folds only and never for validation or testing. They were checked visually by the first author; no plant pathologist has checked them.

## 3. Powdery Mildew prompts

### 3.1 Starting prompt

> Generate a close up image of a sesame leaf plant with powdery mildew, with a background of soil and other plants.

### 3.2 JSON template A (whole plant)

Requested with: *"Can you describe this image in JSON format, so I can use it as a prompt? Also make sure this prompt allows variations of close-ups and far looks, different angles, lightings, etc."*

```json
{
  "prompt_template": {
    "subject": "Sesame plant (Sesamum indicum)",
    "condition": "Severe powdery mildew infection, characterized by white, flour-like fungal patches and dusty spots on the leaf surface",
    "botanical_details": [
      "ovate green leaves",
      "hairy stems",
      "visible seed capsules (pods) along the stalk"
    ],
    "environment": {
      "ground": "dark, tilled organic soil",
      "background": "blurred agricultural field with rows of similar sesame crops",
      "atmosphere": "natural outdoor farm setting"
    },
    "customizable_parameters": {
      "composition": "placeholder_composition",
      "camera_angle": "placeholder_angle",
      "lighting": "placeholder_lighting",
      "image_quality": "photorealistic, macro photography, high resolution, 8k, sharp focus"
    }
  },
  "variation_options": {
    "compositions": [
      "macro close-up",
      "medium shot",
      "wide-angle field view",
      "top-down perspective"
    ],
    "angles": [
      "eye-level",
      "low-angle looking up",
      "side profile",
      "birds-eye view"
    ],
    "lighting_conditions": [
      "diffused overcast daylight",
      "golden hour sunlight",
      "harsh midday sun with sharp shadows",
      "soft morning mist"
    ]
  }
}
```

### 3.3 JSON template B (leaf in focus)

Requested with: *"Okay, remove the wide view, and also sometimes it generates an image of the full plant; I only want to see a close-up of the leaf sometimes, so add that as well."*

```json
{
  "prompt_template": {
    "subject": "Sesame plant leaf (Sesamum indicum)",
    "focus_target": "placeholder_focus",
    "condition": "Heavy powdery mildew infection; dense, white, fuzzy fungal mycelium and flour-like spots covering the green leaf surface",
    "botanical_details": [
      "prominent leaf veins",
      "textured, slightly serrated leaf edges",
      "fine hairs on the leaf and stem"
    ],
    "environment": {
      "background": "blurred dark soil and soft-focus green foliage",
      "depth_of_field": "shallow depth of field to emphasize the fungal texture"
    },
    "customizable_parameters": {
      "composition": "placeholder_composition",
      "camera_angle": "placeholder_angle",
      "lighting": "placeholder_lighting",
      "image_style": "high-detail macro photography, scientific botanical specimen, sharp focus, 8k"
    }
  },
  "variation_options": {
    "focus_targets": [
      "extreme macro of a single infected leaf",
      "close-up focusing on the upper leaf surface",
      "medium shot showing the leaf attached to the main stem"
    ],
    "compositions": [
      "extreme close-up",
      "macro shot",
      "medium-close shot"
    ],
    "angles": [
      "top-down view of the leaf",
      "45-degree side profile",
      "oblique angle to show texture depth"
    ],
    "lighting_conditions": [
      "diffused natural daylight",
      "bright midday sun with high contrast",
      "soft morning light",
      "backlit to highlight fungal fuzz"
    ]
  }
}
```

### 3.4 JSON template C (infection stages)

Requested with: *"Can you add to that JSON that it should also show different stages, like 10% infected, 20%, 50%, 70%, etc.?"*

```json
{
  "prompt_template": {
    "subject": "Sesame plant leaf (Sesamum indicum)",
    "focus_target": "placeholder_focus",
    "infection_severity": "placeholder_severity",
    "visual_progression": "placeholder_visual_description",
    "botanical_details": [
      "prominent leaf veins",
      "textured leaf surface",
      "fine botanical hairs (trichomes)"
    ],
    "environment": {
      "background": "blurred dark soil and soft-focus green foliage",
      "depth_of_field": "shallow depth of field"
    },
    "customizable_parameters": {
      "composition": "placeholder_composition",
      "camera_angle": "placeholder_angle",
      "lighting": "placeholder_lighting",
      "image_style": "high-detail macro photography, scientific botanical specimen, 8k"
    }
  },
  "variation_options": {
    "infection_stages": [
      {
        "percentage": "10% Infected (Early Stage)",
        "description": "Small, isolated white circular spots; mostly healthy green leaf visible."
      },
      {
        "percentage": "30% Infected (Moderate Stage)",
        "description": "Scattered patches of white powdery fungus spreading across the leaf blade."
      },
      {
        "percentage": "50% Infected (Advanced Stage)",
        "description": "Half of the leaf surface covered in a dusty white coating; slight yellowing of the green tissue."
      },
      {
        "percentage": "80% Infected (Severe Stage)",
        "description": "Dense, thick white fungal mycelium covering nearly the entire leaf; leaf edges beginning to curl."
      },
      {
        "percentage": "100% Infected (Terminal Stage)",
        "description": "Total white-out of the leaf surface; host tissue looks necrotic and shriveled underneath the fungus."
      }
    ],
    "focus_targets": [
      "extreme macro of a single infected leaf",
      "close-up of leaf attached to stem",
      "top-down view of leaf cluster"
    ],
    "lighting_conditions": [
      "diffused natural daylight",
      "backlit to show fungal texture",
      "soft morning light"
    ]
  }
}
```

### 3.5 Typical follow-up prompts

- "Generate 4 more images, but not wide-view."
- "Generate 4 more different macro close-ups, only of the leaf, and make them vary in the level of infection: 20%, 30% and 40% infected."
- "Generate 4 more different macro close-ups, 10% powdery mildew." (repeated with 20%, 30% and 40%, "make them slightly different")
- "Okay, generate 4 more images with mild powdery mildew, but a close-up of the leaf itself, and make the background include more plants."
- "4 more, but with different angles and lightings." / "Generate 4 more, but have multiple leaves in the same image."
- "Generate 4 more, but focus on the leaf, and make the leaf 30% powdery mildew; each one has a slightly different shape." (repeated with 20% to 80%, with additions such as "make the background also slightly different", "make the background like a field of other sesame plants", "with a background of other leaves close up too", "different angle and lighting" and "slightly different texture and very slight colour shift")
- "Generate 4 more, focus on the leaf and the entire plant, and make the leaf 40% powdery mildew, with a background of other leaves close up too; make them look like they were captured from different fields."
- "Do not add any labels around the images." / "Generate 4 more, no labels or words, 35% infected."

## 4. Phyllody prompts

### 4.1 Starting prompts

- "Generate a close up image of sesame phyllody infected plant."
- "Generate an image divided into 4 parts, each part shows a different version of sesame plant with phyllody infection, with a background of soil and other sesame plants."
- "Okay, now generate 4 images into 1 image; make sure each one is different."
- "I don't want partial infection. Also do not add any words or text, just give me pure images."
- "These are some samples, can you generate more similar images?" (with three real ERFD Phyllody photos attached)
- "Can you just mainly focus on the phyllody infection?"

### 4.2 JSON description used as a prompt

Gemini was asked to "describe this sesame plant with phyllody infection in JSON"; the result was pasted back with "4 more images following this":

```json
{
  "plant_type": "Sesame (Sesamum indicum)",
  "condition": "Phyllody Infection",
  "causative_agent": "Phytoplasma",
  "visual_observations": {
    "floral_deformation": "The flowers exhibit classic phyllody symptoms where reproductive floral parts are transformed into leaf-like structures (foliaceous carpels).",
    "coloration": "The infected floral organs show a distinct greening (virescence) instead of the normal white, pink, or yellow pigmentation.",
    "morphology": "The central flower in the image is swollen and cup-shaped, failing to develop into a normal tubular corolla. There is a visible proliferation of small, leafy appendages emerging from the floral apex.",
    "stem_and_leaves": "The surrounding leaves appear narrow and lanceolate, while the main stem shows shortened internodes, which often leads to a 'bunchy top' or 'witches' broom' appearance in advanced stages."
  },
  "symptom_classification": [
    "Phyllody (leafy flower development)",
    "Virescence (greening of petals)",
    "Floral proliferation"
  ],
  "agricultural_impact": "The infection renders the plant sterile, preventing the formation of capsules and seeds, leading to significant yield loss."
}
```

### 4.3 Typical follow-up prompts

- "Okay, now create 4 more images in one image; make sure each one is different from each other so they can be used in a dataset."
- "Generate more images, with a more packed background of other sesame plants." / "More, but with a slightly less packed background."
- "4 more images with a close-up. Virescence & Phyllody: shows the transformation of floral parts into green, leafy structures. The 'flowers' lack normal pigmentation and have a thickened, cup-like texture."
- "4 more images of sesame plant with phyllody infection, close-up. Morphological distortion: highlights the abnormal growth at the axils. The structures are distinctly vegetative where reproductive organs should be, providing a clear contrast in texture and shape."
- "Make the images look like realistic field images, and not AI generated."
- "Generate 4 more images, but with a macro close-up." / "... with a medium close-up." / "... with a wider view." / "... with a slightly different background."

### 4.4 Parameterised prompt

At the end of the Phyllody chat, Gemini was asked for one prompt with parameters that can be changed between generations. It returned the following prompt, in which each `{PARAMETER}` is replaced by one of the options in the table below:

```text
A professional 2x2 grid of four distinct, high-resolution field photographs
of Sesame (Sesamum indicum) exhibiting clear symptoms of Phyllody infection.
Each quadrant shows a different plant specimen. {CAMERA_ANGLE} focus on
{SYMPTOM_FOCUS} and {INFECTION_PERCENTAGE} infection. The background
features {BACKGROUND_DENSITY} of healthy sesame rows under
{LIGHTING_CONDITION}. The ground is {SOIL_TYPE}. Authentic agricultural
photography, natural colors, high sharp detail on the virescence and
foliaceous carpels, no text, no watermarks, 8k resolution, realistic
textures.
```

| Parameter | Options (one per generation) |
|---|---|
| `CAMERA_ANGLE` | Macro close-up; medium shot; eye-level close-up; slightly top-down "plan view" |
| `SYMPTOM_FOCUS` | Witches' broom growth; green leaf-like flowers (virescence); thickened stems with axillary proliferation; stunted pods |
| `INFECTION_PERCENTAGE` | 20% early stage; 50% moderate stage; 85% advanced stage; 100% total systemic infection |
| `BACKGROUND_DENSITY` | A dense lush field; a sparse field with visible rows; a blurry bokeh of distant plants |
| `LIGHTING_CONDITION` | Harsh midday sun with shadows; soft overcast daylight; golden hour morning light; hazy afternoon sun |
| `SOIL_TYPE` | Dry cracked clay soil; moist dark fertile earth; sandy loam with small pebbles; dusty brown field dirt |

## Reference

[1] I. Mehta, "Google launches Nano Banana 2 model with faster image generation," TechCrunch, Feb. 26, 2026. https://techcrunch.com/2026/02/26/google-launches-nano-banana-2-model-with-faster-image-generation/
